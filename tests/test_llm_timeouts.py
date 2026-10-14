"""Real ChatOpenRouter client against local servers: a silent one (hang) and a closed port (connection refused)."""
import socket
import threading
import time

from src.errors import LLMConnectionError, LLMTimeoutError, translate_llm_exception
from src.llm import build_chat_model


def _silent_server():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    conns = []

    def run():
        while True:
            try:
                conns.append(srv.accept()[0])  # accept and never answer
            except OSError:
                return

    threading.Thread(target=run, daemon=True).start()
    return srv, srv.getsockname()[1]


def _call(base_url, timeout, retries):
    model = build_chat_model("openrouter/auto", "sk-or-test-key-1234567890", timeout=timeout, max_retries=retries, base_url=base_url)
    start = time.time()
    try:
        model.invoke("hi")
    except Exception as exc:
        return translate_llm_exception(exc), time.time() - start
    raise AssertionError("expected a failure")


def test_server_that_never_answers_times_out_within_the_budget():
    srv, port = _silent_server()
    try:
        err, elapsed = _call(f"http://127.0.0.1:{port}/api/v1", timeout=2, retries=1)
    finally:
        srv.close()
    assert isinstance(err, LLMTimeoutError), err.error_code
    assert elapsed < 10, f"took {elapsed:.1f}s (old behaviour: minutes)"


def test_refused_connection_fails_immediately_as_connection_error():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again -> nothing listens here
    err, elapsed = _call(f"http://127.0.0.1:{port}/api/v1", timeout=30, retries=1)
    assert isinstance(err, LLMConnectionError) and elapsed < 5
