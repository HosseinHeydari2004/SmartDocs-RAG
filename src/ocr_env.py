"""Locate the external OCR tools (Tesseract, Poppler) without requiring them on PATH.

Resolution order for Tesseract: ``TESSERACT_CMD`` env var -> ``tesseract`` on PATH -> well-known install
folders (Windows / macOS Homebrew / Linux). Poppler (needed for scanned PDFs): ``POPPLER_PATH`` env var
(the folder containing ``pdftoppm``) -> PATH.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().strip('"').strip("'").strip()
    return value or None


def _tesseract_candidates() -> list[str]:
    local = os.environ.get("LOCALAPPDATA")
    paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    if local:
        paths.insert(2, str(Path(local) / "Programs" / "Tesseract-OCR" / "tesseract.exe"))
    return paths


def find_tesseract() -> str | None:
    explicit = _clean(os.getenv("TESSERACT_CMD"))
    if explicit:
        return explicit  # honour it even if wrong, so the status can say exactly what is wrong
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _tesseract_candidates():
        if Path(candidate).is_file():
            return candidate
    return None


def configure_tesseract() -> str | None:
    """Point pytesseract at the located executable. Returns the command in use."""
    import pytesseract

    cmd = find_tesseract()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd or pytesseract.pytesseract.tesseract_cmd


def find_poppler_path() -> str | None:
    """Folder to pass to pdf2image, or None when Poppler is on PATH (or absent)."""
    return _clean(os.getenv("POPPLER_PATH"))


def poppler_available() -> bool:
    folder = find_poppler_path()
    if folder:
        return any((Path(folder) / name).is_file() for name in ("pdftoppm", "pdftoppm.exe"))
    return shutil.which("pdftoppm") is not None
