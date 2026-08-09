"""
LangChain RAG - step 2: chain.

Builds a hybrid (dense + keyword) retriever with cross-encoder reranking
over the persisted Chroma store, and an LCEL pipeline (prompt | llm |
parser) for generation. Retrieval and generation stay two explicit steps
so `sources` can be returned alongside the answer without an opaque
RetrievalQA call.
"""

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT
from app.langchain_rag.ingest import build_section_documents


@dataclass(frozen=True)
class GenerationResult:
    """The final answer shown to the user, plus which sections it's based on."""

    answer: str
    sources: list[str]


_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Here are the relevant excerpts from the Consumer Protection Act, 2019:\n\n"
            "{context}\n\n---\n\nUser's question: {query}\n\n"
            "Answer the question using only the excerpts above, following the rules in your "
            "system instructions.",
        ),
    ]
)


def _format_docs(docs: list[Document]) -> str:
    """Render retrieved Documents as labeled, verbatim text blocks the LLM can cite by section."""
    return "\n\n".join(f"[Section {d.metadata['section']}]\n{d.page_content}" for d in docs)


class LangChainRAGChain:
    """
    Wraps a hybrid-search + reranking retriever and an LCEL generation
    pipeline behind one `answer()` call, so the FastAPI route doesn't need
    to know about LangChain internals.
    """

    def __init__(self):
        # Loaded with task_type="retrieval_query" -- the query-side twin of
        # the "retrieval_document" embeddings used at ingest time. Gemini's
        # embedding model is trained asymmetrically, so matching task_type
        # on both ends measurably improves retrieval quality.
        query_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            task_type="retrieval_query",
        )
        vectorstore = Chroma(
            persist_directory=str(settings.chroma_persist_dir),
            embedding_function=query_embeddings,
        )
        # Dense leg: semantic similarity search over the embedded chunks --
        # good at matching meaning/paraphrase, weak on exact statutory
        # wording (e.g. a query containing "spurious goods" verbatim).
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.hybrid_candidate_k})

        # Sparse leg: BM25 keyword search over the SAME chunks (rebuilt from
        # the PDF -- cheap, no embedding calls, since chunking is just
        # regex/string work). Good at exact-term matches dense search can
        # blur together in embedding space.
        bm25_retriever = BM25Retriever.from_documents(build_section_documents())
        bm25_retriever.k = settings.hybrid_candidate_k

        # Hybrid search: merge both candidate lists by weighted score
        # (reciprocal rank fusion under the hood) instead of picking one
        # strategy -- catches cases either leg alone would miss.
        hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[settings.dense_weight, settings.bm25_weight],
        )

        # Reranker: a cross-encoder scores each (query, candidate) pair
        # JOINTLY (unlike embedding similarity, which scores them
        # independently), so it can catch cases where the hybrid merge's
        # heuristic weighting still put a weaker match above a stronger one.
        # Trims the wider hybrid_candidate_k pool back down to the final
        # retrieval_top_k that actually reaches the prompt.
        reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=settings.rerank_model),
            top_n=settings.retrieval_top_k,
        )
        self._retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=hybrid_retriever,
        )

        llm = ChatGoogleGenerativeAI(model=settings.generation_model, google_api_key=settings.google_api_key)
        # LCEL pipeline: fills the prompt template, sends it to Gemini, and
        # StrOutputParser pulls the plain text out of the response object.
        self._generation_chain = _PROMPT | llm | StrOutputParser()

    def answer(self, query: str) -> GenerationResult:
        """
        Retrieve (hybrid search + rerank), then generate a plain-language,
        cited answer from the resulting chunks.

        Two explicit calls (retrieve, then generate) rather than one
        RetrievalQA.invoke(): keeps the retrieved sections available to
        return as `sources` without depending on the model to self-report
        what it used.
        """
        docs = self._retriever.invoke(query)
        answer_text = self._generation_chain.invoke({"context": _format_docs(docs), "query": query})
        sources = [f"Section {d.metadata['section']}" for d in docs]
        return GenerationResult(answer=answer_text, sources=sources)
