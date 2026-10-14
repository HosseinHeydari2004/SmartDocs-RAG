"""The UI can start the backend itself (`streamlit run ui/app.py` alone is enough) - tested with a REAL uvicorn process."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))
import common  # noqa: E402
from api_client import ApiClient  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_start_backend_brings_up_a_working_api(tmp_path):
    url = f"http://127.0.0.1:{_free_port()}"
    env = {**os.environ, "VECTORDB_PATH": str(tmp_path / "db")}
    proc = common.start_backend(url, log_path=tmp_path / "api.log", env=env)
    try:
        client = ApiClient(url)
        assert common.wait_healthy(client, proc, timeout=90), common.read_log_tail(tmp_path / "api.log")
        assert client.health()["status"] == "ok"
        assert client.stats()["chroma_status"] == "ok"
    finally:
        proc.terminate()
        proc.wait(10)


def test_wait_healthy_gives_up_early_when_process_dies(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    start = time.time()
    assert common.wait_healthy(ApiClient(f"http://127.0.0.1:{_free_port()}"), proc, timeout=60) is False
    assert time.time() - start < 10  # did not wait for the full timeout


def test_local_url_detection():
    assert common._is_local("http://localhost:8000") and common._is_local("http://127.0.0.1:9000")
    assert not common._is_local("http://api:8000") and not common._is_local("https://example.com")


def test_read_log_tail_missing_file(tmp_path):
    assert common.read_log_tail(tmp_path / "nope.log") == ""
