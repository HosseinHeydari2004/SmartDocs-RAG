import io

from tests.conftest import txt, wait_job


def test_health_and_docs(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200


def test_unsupported_extension_rejected_with_415(client):
    r = client.post("/api/v1/documents/upload", files=[("files", ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream"))])
    assert r.status_code == 415
    body = r.json()
    assert body["error_code"] == "UNSUPPORTED_FILE_TYPE" and body["request_id"]


def test_oversize_file_rejected_with_413(client):
    big = io.BytesIO(b"a" * (1024 * 1024 + 10))
    r = client.post("/api/v1/documents/upload", files=[("files", ("big.txt", big, "text/plain"))])
    assert r.status_code == 413 and r.json()["error_code"] == "FILE_TOO_LARGE"


def test_spoofed_pdf_rejected(client):
    r = client.post("/api/v1/documents/upload", files=[("files", ("fake.pdf", io.BytesIO(b"not a pdf at all"), "application/pdf"))])
    assert r.status_code == 415 and r.json()["error_code"] == "FILE_CONTENT_MISMATCH"


def test_empty_file_rejected(client):
    r = client.post("/api/v1/documents/upload", files=[("files", ("e.txt", io.BytesIO(b""), "text/plain"))])
    assert r.status_code == 400 and r.json()["error_code"] == "EMPTY_FILE"


def test_path_traversal_name_is_sanitised(client):
    r = client.post("/api/v1/documents/upload", files=[txt("../../etc/passwd.txt")])
    assert r.status_code == 202
    job = wait_job(client, r.json()["id"])
    assert job["files"][0]["filename"] == "passwd.txt"
    assert job["status"] == "completed"


def test_mixed_batch_one_bad_file_does_not_block_others(client):
    files = [txt("good.txt"), ("files", ("bad.exe", io.BytesIO(b"MZ"), "x"))]
    r = client.post("/api/v1/documents/upload", files=files)
    assert r.status_code == 202
    job = wait_job(client, r.json()["id"])
    by_name = {f["filename"]: f for f in job["files"]}
    assert by_name["good.txt"]["status"] == "success"
    assert by_name["bad.exe"]["status"] == "failed" and by_name["bad.exe"]["error_code"] == "UNSUPPORTED_FILE_TYPE"
    assert job["status"] == "completed_with_errors"


def test_validation_error_does_not_echo_secret(client):
    r = client.put("/api/v1/settings/llm", json={"api_key": "short"})
    assert r.status_code == 422
    assert "short" not in r.text
    assert r.json()["error_code"] == "VALIDATION_ERROR"
