"""Request/response models shared across the v1 API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A user's natural-language question about the Act."""

    query: str = Field(..., min_length=1, description="The user's question")


class ChatResponse(BaseModel):
    """A plain-language answer plus the Act section numbers it's based on."""

    answer: str
    sources: list[str]
