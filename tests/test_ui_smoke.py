"""Streamlit pages rendered with AppTest against a REAL uvicorn-served API (fake LLM/embeddings)."""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import requests
import uvicorn
from streamlit.testing.v1 import AppTest

from src.api.main import create_app
from tests.conftest import KEY, SAMPLE_TEXT

UI = Path(__file__).resolve().parent.parent / "ui"
sys.path.insert(0, str(UI))


@pytest.fixture(autouse=True)
def _no_autostart(monkeypatch):
    monkeypatch.setenv("SMARTDOCS_AUTOSTART_API", "0")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_api(container, monkeypatch):
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(container), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("SMARTDOCS_API_URL", base)
    yield base
    server.should_exit = True
    thread.join(5)


def _run(page: str, **kw) -> AppTest:
    return AppTest.from_file(str(UI / "views" / page), default_timeout=30).run(**kw)


def _all_text(at: AppTest) -> str:
    parts = []
    for coll in (at.markdown, at.caption, at.error, at.warning, at.info, at.success, at.metric, at.subheader):
        parts += [str(getattr(e, "value", "")) for e in coll]
    return "\n".join(parts)


def _ingest(base: str, name="a.txt"):
    r = requests.post(base + "/api/v1/documents/upload", files=[("files", (name, SAMPLE_TEXT.encode(), "text/plain"))])
    jid = r.json()["id"]
    for _ in range(200):
        job = requests.get(f"{base}/api/v1/jobs/{jid}").json()
        if job["status"] in ("completed", "failed", "completed_with_errors"):
            return job
        time.sleep(0.05)
    raise AssertionError("ingest timeout")


def test_dashboard_shows_real_state(live_api):
    at = _run("dashboard.py")
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["FastAPI service"] == "Connected" and values["Indexed documents"] == "0"
    job = _ingest(live_api)
    at = _run("dashboard.py")
    values = {m.label: m.value for m in at.metric}
    assert values["Indexed chunks"] == str(job["files"][0]["chunks"])


def test_dashboard_when_api_down_shows_persian_error_not_traceback(monkeypatch):
    monkeypatch.setenv("SMARTDOCS_API_URL", f"http://127.0.0.1:{_free_port()}")
    at = _run("dashboard.py")
    assert not at.exception
    assert any("FastAPI" in e.value for e in at.error)
    assert any("Tried to reach: http://127.0.0.1" in c.value for c in at.caption)  # tells the user WHERE it looked
    assert any("How to fix this" in str(e.label) for e in at.expander)
    assert {m.label: m.value for m in at.metric}["Indexed documents"] == "Unknown"  # unknown, not a fake number


def test_chat_empty_index_guides_to_upload(live_api):
    at = _run("chat.py")
    assert not at.exception
    assert any("Upload a document first" in i.value for i in at.info)


def test_chat_roundtrip_shows_answer_sources_and_filters(live_api, fake_chat):
    from src.schemas.query_filter import QueryMetadataFilter

    _ingest(live_api)
    fake_chat.query_filter = QueryMetadataFilter(year_gte=2020)
    at = _run("chat.py")
    at.chat_input[0].set_value("RAG چگونه کار می‌کند؟").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    text = _all_text(at)
    assert fake_chat.answer in text
    assert "a.txt" in " ".join(str(e.label) for e in at.expander)  # real source shown
    assert any("Detected filters" in str(e.label) for e in at.expander)
    assert any(str(r) for r in at.dataframe)  # applied-filter table rendered
    at.button[0].click().run()  # clear history
    assert len(at.chat_message) == 0


def test_chat_llm_error_is_persian_with_retry(live_api, fake_chat):
    _ingest(live_api)

    class RL(Exception):
        status_code = 429

    fake_chat.raise_on_invoke = RL()
    at = _run("chat.py")
    at.chat_input[0].set_value("سؤال").run()
    assert not at.exception
    assert any("rate limit" in e.value for e in at.error)
    assert any(b.label == "Retry" for b in at.button)


def test_settings_key_form_is_wired_to_backend_and_key_not_rendered(live_api, container):
    at = _run("settings.py")
    assert not at.exception
    new_key = "sk-or-ui-session-key-98765"
    at.text_input[0].set_value(new_key)
    at.text_input[1].set_value("openai/gpt-4o-mini")
    next(b for b in at.button if b.label == "Save settings").click().run()
    assert container.runtime._llm.api_key == new_key and container.runtime._llm.model == "openai/gpt-4o-mini"
    assert new_key not in _all_text(at) and KEY not in _all_text(at)


def test_settings_connection_test_reports_error(live_api, fake_chat):
    class Boom(Exception):
        status_code = 401

    fake_chat.raise_on_invoke = Boom()
    at = _run("settings.py")
    next(b for b in at.button if b.label.startswith("Test connection")).click().run()
    assert any("invalid" in e.value for e in at.error)


def test_documents_page_lists_and_deletes_with_confirmation(live_api):
    _ingest(live_api)
    at = _run("documents.py")
    assert not at.exception
    delete_btn = next(b for b in at.button if b.label == "Delete this document")
    assert delete_btn.disabled  # needs the confirmation checkbox first
    at.checkbox[0].check().run()
    next(b for b in at.button if b.label == "Delete this document").click().run()
    assert requests.get(live_api + "/api/v1/stats").json()["chunk_count"] == 0


def test_upload_page_renders_with_ocr_status(live_api):
    at = _run("upload.py")
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Digital documents", "Scanned documents"]
