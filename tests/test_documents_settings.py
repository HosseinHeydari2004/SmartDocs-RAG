import logging

from src.logging_config import RedactingFilter, redact
from tests.conftest import KEY, txt, wait_job


def _ingest(client, name, text=None):
    f = txt(name) if text is None else txt(name, text)
    return wait_job(client, client.post("/api/v1/documents/upload", files=[f]).json()["id"])


def test_clear_requires_confirmation(client):
    _ingest(client, "a.txt")
    r = client.delete("/api/v1/documents")
    assert r.status_code == 400 and r.json()["error_code"] == "CONFIRMATION_REQUIRED"
    assert client.get("/api/v1/stats").json()["chunk_count"] > 0
    r = client.delete("/api/v1/documents", params={"confirm": "true"})
    assert r.status_code == 200 and r.json()["deleted_chunks"] > 0
    assert client.get("/api/v1/stats").json()["chunk_count"] == 0


def test_delete_single_document_leaves_others(client):
    _ingest(client, "a.txt")
    _ingest(client, "b.txt", "A completely different text. It has other sentences. Nothing alike here at all.")
    docs = client.get("/api/v1/documents").json()["documents"]
    assert len(docs) == 2
    target = next(d for d in docs if d["filename"] == "a.txt")
    r = client.delete(f"/api/v1/documents/{target['doc_id']}")
    assert r.status_code == 200 and r.json()["deleted_documents"] == 1
    left = client.get("/api/v1/documents").json()["documents"]
    assert [d["filename"] for d in left] == ["b.txt"]


def test_delete_unknown_and_malformed_ids(client):
    assert client.delete("/api/v1/documents/" + "0" * 32).status_code == 404
    assert client.delete("/api/v1/documents/not-an-id").status_code == 422


def test_delete_blocked_while_job_active(client, container):
    job = container.jobs.create("digital", ["x.txt"])
    container.jobs.mutate(job.id, lambda j: setattr(j, "status", "running"))
    r = client.delete("/api/v1/documents", params={"confirm": "true"})
    assert r.status_code == 409 and r.json()["error_code"] == "INGESTION_IN_PROGRESS"


def test_job_not_found(client):
    assert client.get("/api/v1/jobs/" + "a" * 32).status_code == 404


def test_settings_status_never_contains_key(client):
    body = client.get("/api/v1/settings/status").text
    assert KEY not in body and "sk-or" not in body
    llm = client.get("/api/v1/settings/status").json()["llm"]
    assert llm["key_configured"] is True and llm["key_source"] == "env"


def test_put_key_updates_runtime_and_never_echoes_it(client, container):
    new_key = "sk-or-session-key-ABCDEFGH"
    r = client.put("/api/v1/settings/llm", json={"api_key": new_key, "model": "openai/gpt-4o-mini"})
    assert r.status_code == 200 and new_key not in r.text
    assert r.json() == {"provider": "openrouter", "model": "openai/gpt-4o-mini", "key_configured": True, "key_source": "session"}
    assert container.runtime._llm.api_key == new_key  # really wired to the runtime
    assert new_key not in client.get("/api/v1/settings/status").text
    r = client.delete("/api/v1/settings/llm/key")
    assert r.json()["key_source"] == "env"


def test_put_requires_something_and_validates_model(client):
    assert client.put("/api/v1/settings/llm", json={}).status_code == 400
    assert client.put("/api/v1/settings/llm", json={"model": "bad model!"}).status_code == 422


def test_llm_test_endpoint_success_and_failure(client, fake_chat):
    assert client.post("/api/v1/settings/llm/test").json()["ok"] is True

    class Boom(Exception):
        status_code = 401

    fake_chat.raise_on_invoke = Boom()
    r = client.post("/api/v1/settings/llm/test")
    assert r.status_code == 401 and r.json()["error_code"] == "LLM_AUTH_FAILED"


def test_unhandled_exception_returns_generic_500(client, container, monkeypatch):
    def boom():
        raise RuntimeError("secret internal path /srv/x")

    monkeypatch.setattr(container.runtime.__class__, "llm_status", lambda self: boom())
    r = client.get("/api/v1/settings/status")
    assert r.status_code == 500
    body = r.json()
    assert body["error_code"] == "INTERNAL_ERROR" and "secret" not in r.text and body["request_id"]


def test_request_id_header_roundtrip(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


def test_redaction():
    assert "sk-or-abcdefgh12345" not in redact("key sk-or-abcdefgh12345 leaked")
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "Authorization: Bearer abcdefghijklmnop", None, None)
    RedactingFilter().filter(rec)
    assert "abcdefghijklmnop" not in rec.msg
