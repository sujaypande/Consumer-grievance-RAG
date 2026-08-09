"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router

app = FastAPI(
    title="Consumer Grievance RAG",
    description="Hybrid-search + reranked RAG (LangChain) over the Consumer Protection Act, 2019.",
    version="0.1.0",
)

# Allows the Vite dev server to call this API from the browser --
# without this, the browser blocks the frontend's fetch calls. Matches
# any localhost port (not just 5173) since Vite auto-picks the next free
# port (5174, 5175, ...) if 5173 is already in use.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    """Liveness check -- why: lets you confirm the server (and its .env/API key) loaded before hitting a real RAG route."""
    return {"status": "ok"}
