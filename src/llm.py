"""Chat-model factory (OpenRouter through LangChain's ``init_chat_model``)."""
from __future__ import annotations

from typing import Any


def build_chat_model(
    model: str, api_key: str, *, timeout: int = 30, max_retries: int = 1, base_url: str | None = None
) -> Any:
    """Build the OpenRouter chat model with predictable time limits.

    Two library pitfalls handled here:
    * ``timeout`` of langchain-openrouter is in MILLISECONDS (passing seconds made every call fail at once);
    * its built-in retry policy retries connection errors for up to ``150 s * max_retries`` (and the SDK default
      is even longer), so a blocked network looked like a hang. We build the SDK client ourselves with a short,
      bounded retry budget and no retries on connection errors (fail fast, tell the user it is a network problem).
    """
    from langchain.chat_models import init_chat_model

    timeout_ms = int(timeout * 1000)
    kwargs: dict[str, Any] = {
        "model_provider": "openrouter",
        "api_key": api_key,
        "temperature": 0,
        "timeout": timeout_ms,
        "max_retries": 0,
    }
    try:
        import openrouter
        from openrouter.utils import BackoffStrategy, RetryConfig

        retry = RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=500,
                max_interval=2000,
                exponent=1.5,
                max_elapsed_time=timeout_ms if max_retries > 0 else 1,  # total retry budget, ms
            ),
            retry_connection_errors=False,
        )
        sdk_kwargs: dict[str, Any] = {"api_key": api_key, "timeout_ms": timeout_ms, "retry_config": retry}
        if base_url:
            sdk_kwargs["server_url"] = base_url
        kwargs["client"] = openrouter.OpenRouter(**sdk_kwargs)
    except ImportError:  # SDK layout changed: fall back to the plain LangChain settings
        kwargs["max_retries"] = max_retries
    return init_chat_model(model, **kwargs)
