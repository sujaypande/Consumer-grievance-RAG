"""
LangChain RAG - step 1: ingest.

Loads the Act's PDF, splits it into section-aware chunks, embeds them, and
persists them to a local Chroma vectorstore.
"""

import re
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings

# Chroma's own bulk embedding call would send ~100 chunks in one request,
# which immediately exceeds the free tier's 100-per-minute quota. Adding
# documents one at a time with a small delay avoids that entirely.
EMBED_CALL_DELAY_SECONDS = 0.7

# Marks where the Table of Contents / preamble notification text ends and
# the actual numbered Act body begins. Everything before this is TOC
# entries (which look like "34. Jurisdiction of District Commission." --
# indistinguishable from a real section start by pattern alone) and
# amendment-history footnotes, neither of which should become chunks.
BODY_START_MARKER = "BE it enacted by Parliament"

# Matches the start of a numbered section, e.g. "\n34. Jurisdiction of...".
# Requires the text right after the number+period to start with a capital
# letter (a title), which is true for every real section heading but not
# for footnote text like "1. Ins. by Act 33 of 2021...".
SECTION_HEADING_RE = re.compile(r"\n(\d{1,3})\.\s+(?=[A-Z])")

# Sections above this length get split further, on their numbered
# sub-clauses e.g. "(1) ... (2) ...". Only Section 2 (Definitions, ~40
# separate definitions in one section) is long enough to need this in
# practice -- without it, one chunk would try to represent 40 unrelated
# definitions at once, diluting embedding similarity for any single one.
LONG_SECTION_CHAR_THRESHOLD = 1800
SUBCLAUSE_RE = re.compile(r"\n\((\d{1,3})\)\s+")


def load_pdf_text() -> str:
    """Load the Act PDF via LangChain's PyPDFLoader and concatenate its pages into one string."""
    pages = PyPDFLoader(str(settings.source_pdf_path)).load()
    return "\n".join(page.page_content for page in pages)


def split_into_section_chunks(full_text: str) -> list[dict]:
    """
    Turn the raw Act text into a list of {section, text} chunks, one per
    section (or per sub-clause, for long sections like Definitions).

    WHY section-aware instead of fixed-size chunking: legal text already has
    a natural semantic unit -- the section. Splitting at those boundaries
    keeps each chunk topically coherent (one chunk = one rule or one
    definition) and gives retrieval a citation to point to ("Section 34")
    for free, instead of an arbitrary character window.

    WHY sequential-number validation instead of trusting every regex hit:
    the Act's PDF also contains amendment-history footnotes formatted like
    "1. Ins. by Act 33 of 2021..." which match the same "N. Capitalized..."
    pattern as a real section heading. Real sections are numbered 1..107
    with no gaps, so a match is only accepted if its number is exactly one
    more than the last accepted section -- this rejects footnote noise
    without a hand-maintained blocklist.
    """
    body_start = full_text.find(BODY_START_MARKER)
    body = full_text[body_start:] if body_start != -1 else full_text

    matches = list(SECTION_HEADING_RE.finditer(body))
    accepted = []
    last_section_number = 0
    for match in matches:
        number = int(match.group(1))
        if number == last_section_number + 1:
            accepted.append(match)
            last_section_number = number

    chunks = []
    for i, match in enumerate(accepted):
        section_number = match.group(1)
        start = match.end()
        end = accepted[i + 1].start() if i + 1 < len(accepted) else len(body)
        section_text = body[start:end].strip()

        if len(section_text) <= LONG_SECTION_CHAR_THRESHOLD:
            chunks.append({"section": section_number, "text": section_text})
            continue

        # Long section: split further on its top-level numbered sub-clauses.
        sub_matches = list(SUBCLAUSE_RE.finditer(section_text))
        if len(sub_matches) < 2:
            # No usable sub-clause structure found -- fall back to one chunk
            # rather than silently dropping text.
            chunks.append({"section": section_number, "text": section_text})
            continue

        for j, sub_match in enumerate(sub_matches):
            sub_number = sub_match.group(1)
            sub_start = sub_match.start()
            sub_end = sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(section_text)
            chunks.append(
                {
                    "section": f"{section_number}({sub_number})",
                    "text": section_text[sub_start:sub_end].strip(),
                }
            )

    return chunks


def build_section_documents() -> list[Document]:
    """
    Split the Act text into section-aware chunks and wrap each as a
    LangChain Document.

    Also used by chain.py to build the BM25 (keyword) retriever's corpus --
    calling this again there is cheap (regex/string work only, no embedding
    API calls), so there's no need to persist the chunk list separately the
    way FAISS-based pipelines need a sidecar metadata file.
    """
    full_text = load_pdf_text()
    chunks = split_into_section_chunks(full_text)
    return [Document(page_content=c["text"], metadata={"section": c["section"]}) for c in chunks]


def build_and_persist_vectorstore(force: bool = False) -> Chroma:
    """
    Run the full ingest pipeline (load -> split -> embed -> store) and
    persist the result to backend/data/chroma_db/, unless it already exists.
    """
    persist_dir = str(settings.chroma_persist_dir)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        task_type="retrieval_document",
    )

    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    # Check the actual collection count, not just "does the directory
    # exist" -- a prior run that failed partway through (e.g. hit a rate
    # limit) can leave an empty-but-present Chroma directory behind, which
    # would otherwise be mistaken for a completed ingest.
    if vectorstore._collection.count() > 0 and not force:
        print(f"Chroma store already has {vectorstore._collection.count()} chunks, skipping ingest.")
        return vectorstore

    print("Loading + splitting PDF into section-aware Documents...")
    documents = build_section_documents()
    print(f"Produced {len(documents)} chunks.")

    print("Embedding + storing in Chroma (one chunk at a time, rate-limit paced)...")
    for i, doc in enumerate(documents):
        vectorstore.add_documents([doc])
        if (i + 1) % 20 == 0:
            print(f"  embedded {i + 1}/{len(documents)} chunks...")
        time.sleep(EMBED_CALL_DELAY_SECONDS)

    print(f"Saved {len(documents)} embedded chunks to {persist_dir}.")
    return vectorstore


if __name__ == "__main__":
    build_and_persist_vectorstore()
