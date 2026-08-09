"""Aggregates the v1 API's route groups under their spec'd prefixes."""

from fastapi import APIRouter

from app.api.v1 import langchain_chat

api_router = APIRouter()
api_router.include_router(langchain_chat.router, prefix="/langchain", tags=["langchain-rag"])
