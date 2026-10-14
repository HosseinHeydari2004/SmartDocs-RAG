from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from src.api.deps import Container, get_container
from src.api.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="RAG question answering")
def chat(body: ChatRequest, c: Container = Depends(get_container)) -> ChatResponse:
    """Extract metadata filters from the question (LLM), filter + search Chroma, answer from the sources."""
    result = c.rag.answer(
        body.question,
        history=[t.model_dump() for t in body.history],
        k=body.k,
        auto_filter=body.auto_filter,
        manual_filter=body.manual_filter,
    )
    return ChatResponse(**asdict(result))
