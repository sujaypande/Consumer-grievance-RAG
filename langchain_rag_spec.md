# Consumer Grievance RAG — LangChain Implementation

## Goal
Answers questions about the Consumer Protection Act, 2019, built using
LangChain: hybrid (dense + BM25) retrieval, cross-encoder reranking, then
generation.

This is the only RAG pipeline in the live backend. An earlier hand-rolled
("naive," no framework) version was archived to `manual-rag/` at the repo
root once this pipeline moved past naive retrieve-then-generate -- see
`manual-rag/COMPARISON.md` for the original framework-vs-no-framework
writeup, which still holds for the parts that didn't change (chunking,
prompt, response shape).

## Source Document
Same single document: data/consumer_protection_act_2019.pdf

## Architecture

app/langchain_rag/
├── ingest.py    — LangChain document loader + text splitter + Chroma vectorstore
└── chain.py      — RetrievalQA chain or LCEL pipeline

### ingest.py
- Load the Act using a LangChain document loader (PyPDFLoader or load pre-extracted
  text via TextLoader, whichever is cleaner given the source format).
- Split using RecursiveCharacterTextSplitter, OR a custom section-aware splitter
  mirroring the manual project's chunking logic (recommended — makes the
  comparison table meaningful, since both projects then chunk the same way and
  only the plumbing around it differs).
- Embed and store in Chroma, persisted locally to backend/data/chroma_db/.
- Same embedding model as the manual project, so retrieval quality is comparable.

### chain.py
- Build a RetrievalQA chain or an LCEL pipeline: retriever | prompt | llm | parser.
- Use the SAME plain-language system prompt template as the manual project
  (copy the prompt text exactly — only the plumbing differs, not the instructions
  given to the model).
- Return answer + cited section numbers, same response shape as the manual version.

## API Routes (mounted under /api/v1/langchain)
POST /api/v1/langchain/chat
  Request:  {"query": "string"}
  Response: {"answer": "string", "sources": ["Section 34", "Section 2(7)"]}

## Plain-language requirement
Identical standard to the manual project — same system prompt, same rule: explain
legal terms in plain words, cite sections, keep source chunk text verbatim for
citation accuracy.

## Requirements
- No Docker, Kafka, Redis, or microservices for this phase
- Every function needs a docstring: what it does and why
- Comment which LangChain abstraction is used and what manual step it replaces,
  e.g.: "# RetrievalQA chain replaces manual retriever.py + generator.py from the
  naive implementation"

## Deliverable: comparison table
After both implementations work, write COMPARISON.md at the project root mapping
each RAG step to its manual vs. LangChain implementation:

| Step | Manual (naive) | LangChain |
|------|-----------------|-----------|
| Chunking | ... | ... |
| Embedding + storage | ... | ... |
| Retrieval | ... | ... |
| Prompt assembly | ... | ... |
| Orchestration | ... | ... |
| LLM call | ... | ... |

This table is as important as the code — it's the artifact that demonstrates
understanding of what the framework abstracts, not just that both versions work.

## Retrieval pipeline (hybrid search + reranking)
- Dense leg: Chroma vectorstore, top `hybrid_candidate_k` by embedding similarity
- Sparse leg: `BM25Retriever` over the same section-aware chunks, top `hybrid_candidate_k` by keyword score
- Merge: `EnsembleRetriever` combines both legs by weighted score (`dense_weight`/`bm25_weight`)
- Rerank: a local cross-encoder (`CrossEncoderReranker` + `HuggingFaceCrossEncoder`)
  scores the merged pool query-doc-jointly and cuts it back down to `retrieval_top_k`,
  which is what actually reaches the prompt

## Out of scope
- Auth, user accounts, chat history persistence
- Multi-turn conversation memory
- Query rewriting/expansion, agentic behavior