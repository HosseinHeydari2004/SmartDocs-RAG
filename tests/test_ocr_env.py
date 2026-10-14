from src import ocr_env
from src.services.ocr_status import ocr_status


def test_explicit_env_wins_even_if_path_has_tesseract(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", '"C:\\Custom\\tesseract.exe"')
    monkeypatch.setattr(ocr_env.shutil, "which", lambda name: "/usr/bin/tesseract")
    assert ocr_env.find_tesseract() == "C:\\Custom\\tesseract.exe"  # quotes stripped


def test_falls_back_to_well_known_install_folder_when_not_on_path(monkeypatch, tmp_path):
    fake = tmp_path / "tesseract.exe"
    fake.write_text("x")
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr_env.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_env, "_tesseract_candidates", lambda: ["/nope/tesseract", str(fake)])
    assert ocr_env.find_tesseract() == str(fake)


def test_nothing_found_returns_none(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr_env.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_env, "_tesseract_candidates", lambda: [])
    assert ocr_env.find_tesseract() is None


def test_status_explains_why_when_executable_is_wrong(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/definitely/not/here/tesseract")
    st = ocr_status()
    assert st["available"] is False and st["languages"] == []
    assert "TESSERACT_CMD" in st["reason"] and "/definitely/not/here" in st["reason"]


def test_status_available_when_command_is_valid(monkeypatch):
    import shutil

    real = shutil.which("tesseract")
    if not real:
        import pytest

        pytest.skip("tesseract not installed")
    monkeypatch.setenv("TESSERACT_CMD", real)  # works even though not relying on PATH
    st = ocr_status()
    assert st["available"] is True and st["reason"] is None and st["command"] == real


def test_poppler_path_env(monkeypatch, tmp_path):
    (tmp_path / "pdftoppm.exe").write_text("x")
    monkeypatch.setenv("POPPLER_PATH", str(tmp_path))
    assert ocr_env.find_poppler_path() == str(tmp_path) and ocr_env.poppler_available()
    monkeypatch.setenv("POPPLER_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(ocr_env.shutil, "which", lambda n: None)
    assert not ocr_env.poppler_available()
