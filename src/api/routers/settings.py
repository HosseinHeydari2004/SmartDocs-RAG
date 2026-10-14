from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import Container, get_container
from src.api.schemas import LlmSettingsUpdate, LlmStatus, LlmTestResponse
from src.errors import BadRequestError

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@router.put("/llm", response_model=LlmStatus, summary="Set OpenRouter key / model for this server process")
def update_llm(body: LlmSettingsUpdate, c: Container = Depends(get_container)) -> LlmStatus:
    """The key is kept **in memory only** (lost on restart) and is never returned or logged."""
    if body.api_key is None and body.model is None:
        raise BadRequestError("Provide api_key and/or model.", error_code="NOTHING_TO_UPDATE")
    c.runtime.configure_llm(
        api_key=body.api_key.get_secret_value() if body.api_key else None, model=body.model
    )
    return LlmStatus(**c.runtime.llm_status())


@router.delete("/llm/key", response_model=LlmStatus, summary="Forget the session key (falls back to env)")
def clear_key(c: Container = Depends(get_container)) -> LlmStatus:
    c.runtime.clear_session_key()
    return LlmStatus(**c.runtime.llm_status())


@router.post("/llm/test", response_model=LlmTestResponse, summary="Test the OpenRouter connection with one tiny request")
def test_llm(c: Container = Depends(get_container)) -> LlmTestResponse:
    c.runtime.test_llm()
    return LlmTestResponse(ok=True, model=c.runtime.llm_status()["model"])
