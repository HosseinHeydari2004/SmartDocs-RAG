from __future__ import annotations

from typing import Any

from src.ocr_env import configure_tesseract, poppler_available

_HINT = (
    "Install Tesseract and either add its folder to PATH or set TESSERACT_CMD in .env "
    "(e.g. TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe), then restart the app."
)


def ocr_status() -> dict[str, Any]:
    """Report what OCR can really do on this machine (nothing is assumed), and why not if it cannot."""
    status: dict[str, Any] = {
        "engine": "tesseract",
        "available": False,
        "version": None,
        "languages": [],
        "pdf_supported": poppler_available(),
        "command": None,
        "reason": None,
    }
    try:
        import pytesseract

        status["command"] = configure_tesseract()
        status["version"] = str(pytesseract.get_tesseract_version())
        status["languages"] = sorted(l for l in pytesseract.get_languages(config="") if l != "osd")
        if not status["languages"]:
            status["reason"] = "Tesseract runs but no language data (tessdata) was found."
        else:
            status["available"] = True
    except Exception as exc:
        name = type(exc).__name__
        if name in ("TesseractNotFoundError", "FileNotFoundError", "PermissionError"):
            status["reason"] = f"The Tesseract executable could not be run ({status['command'] or 'not found'}). {_HINT}"
        else:
            status["reason"] = f"Tesseract check failed: {str(exc).splitlines()[0][:200] if str(exc) else name}"
    return status
