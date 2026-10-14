from unittest.mock import patch

from src.errors import LLMConnectionError, translate_llm_exception
from src.llm import build_chat_model
from tests.conftest import txt, wait_job


def test_timeout_is_converted_to_milliseconds_for_the_real_openrouter_client():
    """Regression: 60 (seconds) used to be passed as 60 ms -> instant ConnectTimeout on every call."""
    model = build_chat_model("openrouter/auto", "sk-or-test-key-1234567890", timeout=30)  # builds a REAL ChatOpenRouter
    assert type(model).__name__ == "ChatOpenRouter"
    assert model.request_timeout == 30_000


def test_connect_timeout_maps_to_connection_error_even_when_wrapped():
    class ConnectTimeout(Exception):
        pass

    assert isinstance(translate_llm_exception(ConnectTimeout()), LLMConnectionError)
    try:
        try:
            raise ConnectTimeout("inner")
        except ConnectTimeout as inner:
            raise RuntimeError("sdk wrapper") from inner
    except RuntimeError as wrapped:
        assert isinstance(translate_llm_exception(wrapped), LLMConnectionError)


def test_chat_fails_fast_on_connection_error_without_loading_embeddings_or_calling_generation(client, container, fake_chat):
    wait_job(client, client.post("/api/v1/documents/upload", files=[txt()]).json()["id"])

    class ConnectTimeout(Exception):
        pass

    fake_chat.query_filter = ConnectTimeout()
    fake_chat.prompts.clear()
    with patch.object(container.runtime.vector_store, "search_with_scores", side_effect=AssertionError("must not search")):
        r = client.post("/api/v1/chat", json={"question": "anything"})
    assert r.status_code == 502 and r.json()["error_code"] == "LLM_CONNECTION_FAILED"
    assert fake_chat.prompts == []  # generation was never attempted


def test_embedding_preload_runs_in_background(container):
    container.runtime.preload_embeddings().join(5)
    assert container.runtime.embeddings.loaded
