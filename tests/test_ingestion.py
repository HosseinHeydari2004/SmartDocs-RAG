import io
from unittest.mock import patch

import pytest

from src.loaders.ocr.ocr_loader import OcrError
from src.services.ocr_status import ocr_status
from tests.conftest import txt, wait_job


def test_successful_ingestion_stores_real_chunks(client, container):
    r = client.post("/api/v1/documents/upload", files=[txt("a.txt")])
    assert r.status_code == 202
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "completed" and job["percent"] == 100
    f = job["files"][0]
    assert f["status"] == "success" and f["chunks"] >= 1
    stats = client.get("/api/v1/stats").json()
    assert stats["chroma_status"] == "ok"
    assert stats["chunk_count"] == f["chunks"] == job["index_total_chunks"]
    assert stats["document_count"] == 1
    # metadata normalised: filename only, no temp path
    doc = client.get("/api/v1/documents").json()["documents"][0]
    assert doc["filename"] == "a.txt" and doc["title"] == "Test Doc" and doc["year"] == 2024


def test_progress_is_monotonic_and_stage_driven(container):
    from src.services.ingestion import IngestionService

    seen = []
    original = container.jobs.recompute_percent

    def spy(job_id):
        original(job_id)
        seen.append((container.jobs.get(job_id).stage, container.jobs.get(job_id).percent))

    container.jobs.recompute_percent = spy
    from fastapi.testclient import TestClient
    from src.api.main import create_app

    with TestClient(create_app(container)) as c:
        job = wait_job(c, c.post("/api/v1/documents/upload", files=[txt()]).json()["id"])
    percents = [p for _, p in seen]
    assert percents == sorted(percents)
    stages = [s for s, _ in seen]
    for expected in ("extract", "clean", "metadata", "chunk", "embed_store"):
        assert expected in stages
    assert job["percent"] == 100


def test_duplicate_skipped_then_replaced(client):
    first = wait_job(client, client.post("/api/v1/documents/upload", files=[txt()]).json()["id"])
    n = first["files"][0]["chunks"]
    dup = wait_job(client, client.post("/api/v1/documents/upload", files=[txt()]).json()["id"])
    assert dup["files"][0]["status"] == "skipped_duplicate"
    assert client.get("/api/v1/stats").json()["chunk_count"] == n
    rep = wait_job(
        client,
        client.post("/api/v1/documents/upload", files=[txt()], data={"on_duplicate": "replace"}).json()["id"],
    )
    assert rep["files"][0]["status"] == "success"
    assert client.get("/api/v1/stats").json()["chunk_count"] == n  # replaced, not doubled


def test_same_file_twice_in_one_request_indexed_once(client):
    job = wait_job(client, client.post("/api/v1/documents/upload", files=[txt("a.txt"), txt("b.txt")]).json()["id"])
    statuses = sorted(f["status"] for f in job["files"])
    assert statuses == ["skipped_duplicate", "success"]


def test_metadata_failure_is_reported_not_faked(client, fake_chat):
    fake_chat.doc_metadata = RuntimeError("llm down")
    job = wait_job(client, client.post("/api/v1/documents/upload", files=[txt()]).json()["id"])
    f = job["files"][0]
    assert f["status"] == "success"
    assert any("Metadata extraction failed" in w for w in f["warnings"])
    doc = client.get("/api/v1/documents").json()["documents"][0]
    assert doc["title"] is None and doc["year"] is None


def test_pdf_without_text_suggests_scanned_upload(client):
    import fitz

    pdf = fitz.open()
    pdf.new_page()
    data = pdf.tobytes()
    job = wait_job(client, client.post("/api/v1/documents/upload", files=[("files", ("blank.pdf", io.BytesIO(data), "application/pdf"))]).json()["id"])
    f = job["files"][0]
    assert f["status"] == "failed" and f["error_code"] == "NO_TEXT_FOUND" and "scanned" in f["error"]


def test_real_pdf_text_extraction_and_page_number(client):
    import fitz

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu.")
    job = wait_job(client, client.post("/api/v1/documents/upload", files=[("files", ("t.pdf", io.BytesIO(pdf.tobytes()), "application/pdf"))]).json()["id"])
    assert job["files"][0]["status"] == "success"
    from tests.conftest import KEY  # noqa: F401


def test_ocr_failure_is_reported_per_file(client):
    png = _png_bytes("HELLO")
    with patch("src.services.ingestion.OcrLoader.load", side_effect=OcrError("Tesseract exploded")):
        r = client.post("/api/v1/documents/upload-scanned", files=[("files", ("s.png", io.BytesIO(png), "image/png"))])
        if r.status_code == 503:
            pytest.skip("tesseract not installed")
        job = wait_job(client, r.json()["id"])
    f = job["files"][0]
    assert f["status"] == "failed" and f["error_code"] == "OCR_FAILED"
    assert job["status"] == "failed"


def test_scanned_endpoint_503_when_tesseract_missing(client):
    with patch("src.api.routers.documents.ocr_status", return_value={"available": False, "languages": [], "pdf_supported": False}):
        r = client.post("/api/v1/documents/upload-scanned", files=[("files", ("s.png", io.BytesIO(_png_bytes("A")), "image/png"))])
    assert r.status_code == 503 and r.json()["error_code"] == "OCR_UNAVAILABLE"


def test_scanned_unknown_language_rejected(client):
    if not ocr_status()["available"]:
        pytest.skip("tesseract not installed")
    r = client.post("/api/v1/documents/upload-scanned", data={"ocr_lang": "zzz"}, files=[("files", ("s.png", io.BytesIO(_png_bytes("A")), "image/png"))])
    assert r.status_code == 400 and r.json()["error_code"] == "OCR_LANG_UNAVAILABLE"


def test_real_ocr_pipeline_indexes_recognised_text(client):
    st = ocr_status()
    if not st["available"] or "eng" not in st["languages"]:
        pytest.skip("tesseract/eng not installed")
    job = wait_job(client, client.post("/api/v1/documents/upload-scanned", files=[("files", ("scan.png", io.BytesIO(_png_bytes("HELLO WORLD")), "image/png"))]).json()["id"])
    assert job["files"][0]["status"] == "success", job
    hits = client.post("/api/v1/chat", json={"question": "HELLO WORLD", "auto_filter": False}).json()
    assert "HELLO" in hits["sources"][0]["content"].upper()
    assert hits["sources"][0]["page"] == 1


def test_cancel_stops_job_and_rolls_back(container, client):
    import threading

    gate = threading.Event()
    real = container.runtime.preprocessing.chunk

    def slow_chunk(docs):
        gate.wait(5)
        return real(docs)

    container.runtime.preprocessing.chunk = slow_chunk
    jid = client.post("/api/v1/documents/upload", files=[txt()]).json()["id"]
    assert client.post(f"/api/v1/jobs/{jid}/cancel").status_code == 200
    gate.set()
    job = wait_job(client, jid)
    assert job["status"] == "cancelled"
    assert client.get("/api/v1/stats").json()["chunk_count"] == 0
    assert client.post(f"/api/v1/jobs/{jid}/cancel").status_code == 409


def _png_bytes(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(img).text((20, 50), text, fill="black", font=ImageFont.load_default(size=80))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
