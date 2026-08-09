"""LangChain RAG chat route, backed by app/langchain_rag/chain.py."""

from fastapi import APIRouter

from app.api.v1.schemas import ChatRequest, ChatResponse
from app.langchain_rag.chain import LangChainRAGChain

router = APIRouter()

# Built once when this module is first imported -- its constructor opens
# the persisted Chroma store and builds the LCEL pipeline, both pointless
# to redo on every request.
chain = LangChainRAGChain()


@router.post("/chat", response_model=ChatResponse)
def langchain_chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a question about the Act using the LangChain RAG pipeline:
    Chroma retriever -> LCEL (prompt | llm | parser) generation.
    """
    result = chain.answer(request.query)
    return ChatResponse(answer=result.answer, sources=result.sources)
