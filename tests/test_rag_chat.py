from src.errors import LLMAuthError
from src.schemas.query_filter import QueryMetadataFilter
from tests.conftest import txt, wait_job


def _ingest(client, name="a.txt"):
    return wait_job(client, client.post("/api/v1/documents/upload", files=[txt(name)]).json()["id"])


def _chat(client, **body):
    body.setdefault("question", "How does retrieval augmented generation work?")
    return client.post("/api/v1/chat", json=body)


def test_empty_index_returns_409(client):
    r = _chat(client)
    assert r.status_code == 409 and r.json()["error_code"] == "INDEX_EMPTY"


def test_answer_with_real_sources_and_scores(client, fake_chat):
    _ingest(client)
    r = _chat(client)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["answer"] == fake_chat.answer
    src = body["sources"][0]
    assert src["filename"] == "a.txt" and 0.0 <= src["similarity"] <= 1.0 and src["content"]
    assert "[1]" in str(fake_chat.prompts[-1])  # sources were numbered in the prompt
    assert body["filter_status"] == "none_detected" and body["applied_filter"] == {} and body["chroma_where"] is None


def test_extracted_filter_is_applied_to_chroma(client, fake_chat):
    _ingest(client)  # indexed with year=2024, language=en, type=report
    fake_chat.query_filter = QueryMetadataFilter(year_gte=2020, language="en")
    body = _chat(client).json()
    assert body["filter_status"] == "extracted"
    assert body["extracted_filter"] == {"year_gte": 2020, "language": "en"}
    assert body["chroma_where"] == {"$and": [{"language": {"$eq": "en"}}, {"year": {"$gte": 2020}}]}
    assert body["status"] == "answered"


def test_filter_that_matches_nothing_reports_no_evidence_without_calling_llm(client, fake_chat):
    _ingest(client)
    fake_chat.query_filter = QueryMetadataFilter(year=1999)
    n_before = len(fake_chat.prompts)
    body = _chat(client).json()
    assert body["status"] == "no_evidence" and body["sources"] == []
    assert body["applied_filter"] == {"year": 1999}
    assert len(fake_chat.prompts) == n_before  # no fabricated answer


def test_manual_filter_merges_with_extracted(client, fake_chat):
    _ingest(client)
    fake_chat.query_filter = QueryMetadataFilter(year_gte=2020)
    body = _chat(client, manual_filter={"filename": "a.txt"}).json()
    assert body["applied_filter"] == {"filename": "a.txt", "year_gte": 2020}
    assert body["status"] == "answered"
    body2 = _chat(client, manual_filter={"filename": "other.pdf"}).json()
    assert body2["status"] == "no_evidence"


def test_auto_filter_off_skips_extraction(client, fake_chat):
    _ingest(client)
    fake_chat.query_filter = QueryMetadataFilter(year=1999)
    body = _chat(client, auto_filter=False).json()
    assert body["filter_status"] == "skipped" and body["status"] == "answered"


def test_filter_extraction_failure_is_reported_not_faked(client, fake_chat):
    _ingest(client)
    fake_chat.query_filter = ValueError("bad json")
    body = _chat(client).json()
    assert body["filter_status"] == "failed" and body["filter_error"]
    assert body["applied_filter"] == {} and body["status"] == "answered"


def test_invalid_llm_filter_output_is_failure(client, fake_chat):
    _ingest(client)
    try:
        QueryMetadataFilter(year_gte=2030, year_lte=2000)
    except Exception as e:
        fake_chat.query_filter = e
    assert _chat(client).json()["filter_status"] == "failed"


def test_llm_auth_error_maps_to_401(client, fake_chat):
    _ingest(client)

    class Boom(Exception):
        status_code = 401

    fake_chat.raise_on_invoke = Boom("nope sk-or-secret-value-123456")
    r = _chat(client)
    assert r.status_code == 401 and r.json()["error_code"] == "LLM_AUTH_FAILED"
    assert "secret" not in r.text


def test_rate_limit_maps_to_429(client, fake_chat):
    _ingest(client)

    class RL(Exception):
        status_code = 429

    fake_chat.raise_on_invoke = RL()
    assert _chat(client).status_code == 429


def test_timeout_maps_to_504(client, fake_chat):
    _ingest(client)

    class ReadTimeout(Exception):
        pass

    fake_chat.raise_on_invoke = ReadTimeout()
    assert _chat(client).status_code == 504


def test_missing_key_returns_503(client, container):
    _ingest(client)
    container.runtime._llm.api_key = None
    container.runtime._chat_cache = None
    r = _chat(client)
    assert r.status_code == 503 and r.json()["error_code"] == "LLM_NOT_CONFIGURED"


def test_history_is_forwarded_and_trimmed(client, fake_chat):
    _ingest(client)
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    _chat(client, history=hist)
    msgs = fake_chat.prompts[-1]
    assert len(msgs) == 1 + 6 + 1  # system + max_chat_history + question


def test_validation_blank_question(client):
    assert _chat(client, question="   ").status_code == 422
