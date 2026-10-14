"""Shared UI helpers: styling, English texts, error rendering."""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from api_client import ApiClient, ApiError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_LOG = PROJECT_ROOT / "api.log"

APP_NAME = "SmartDocs-RAG"

ERROR_TEXT: dict[str, str] = {
    "API_UNREACHABLE": "Cannot reach the FastAPI backend. Make sure it is running and that SMARTDOCS_API_URL points to it.",
    "API_TIMEOUT": "The backend took too long to answer. Please try again shortly.",
    "UNSUPPORTED_FILE_TYPE": "This file type is not supported.",
    "FILE_TOO_LARGE": "The file exceeds the maximum allowed size.",
    "FILE_CONTENT_MISMATCH": "The file content does not match its extension.",
    "EMPTY_FILE": "The file is empty.",
    "TOO_MANY_FILES": "Too many files in one request.",
    "NO_FILES": "No files were selected.",
    "NO_TEXT_FOUND": "No extractable text was found. If this is a scanned file, use the Scanned tab.",
    "EXTRACTION_FAILED": "The file could not be read (it may be corrupt or encrypted).",
    "OCR_FAILED": "OCR failed for this file.",
    "OCR_NO_TEXT": "OCR recognised no text in this file.",
    "OCR_UNAVAILABLE": "The OCR engine (Tesseract) is not available on the server.",
    "OCR_LANG_UNAVAILABLE": "The selected OCR language is not installed on the server.",
    "INDEX_EMPTY": "The index is empty. Upload a document first.",
    "EMBEDDING_MODEL_ERROR": "The embedding model could not be loaded. Check the server log.",
    "EMBEDDING_OOM": "Not enough memory for embedding. Lower EMBEDDING_BATCH_SIZE or set EMBEDDING_DEVICE=cpu.",
    "EMBEDDING_MODEL_MISMATCH": "The embedding model differs from the one used to build the index. Restore the old model, or clear the collection and re-index.",
    "VECTOR_STORE_ERROR": "The vector database (Chroma) reported an error. Check the server log.",
    "LLM_NOT_CONFIGURED": "No OpenRouter API key is configured. Add one on the Settings page.",
    "LLM_AUTH_FAILED": "The OpenRouter API key is invalid or not authorised.",
    "LLM_RATE_LIMITED": "The LLM provider rate limit was reached. Try again in a moment.",
    "LLM_CONNECTION_FAILED": "Could not connect to OpenRouter. Check your internet connection, firewall/proxy, VPN or regional blocking (this is a network problem, not your API key).",
    "LLM_TIMEOUT": "The LLM provider did not respond in time.",
    "LLM_UNAVAILABLE": "The LLM provider returned an error or is unavailable.",
    "INGESTION_IN_PROGRESS": "A processing job is running. Wait for it to finish or cancel it.",
    "DUPLICATE": "This exact file is already indexed.",
    "METADATA_EXTRACTION_FAILED": "Metadata extraction failed.",
    "VALIDATION_ERROR": "The input is not valid.",
    "DOCUMENT_NOT_FOUND": "Document not found.",
    "JOB_NOT_FOUND": "Job not found (job state is kept in server memory only).",
    "INTERNAL_ERROR": "Internal server error. Details are in the server log.",
}
TRANSIENT = {"API_UNREACHABLE", "API_TIMEOUT", "LLM_RATE_LIMITED", "LLM_TIMEOUT", "LLM_UNAVAILABLE", "LLM_CONNECTION_FAILED"}

STAGE_TEXT = {
    "receive": "Receiving and validating files",
    "extract": "Extracting text / running OCR",
    "clean": "Cleaning and preprocessing",
    "metadata": "Extracting metadata",
    "chunk": "Chunking",
    "embed_store": "Generating embeddings and storing in Chroma",
    "done": "Finished",
}
FILE_STATUS = {
    "pending": "Pending", "processing": "Processing", "success": "Success",
    "failed": "Failed", "skipped_duplicate": "Duplicate (skipped)", "cancelled": "Cancelled",
}
JOB_STATUS = {
    "queued": "Queued", "running": "Running", "cancelling": "Cancelling",
    "completed": "Completed", "completed_with_errors": "Completed with errors", "failed": "Failed", "cancelled": "Cancelled",
}
TERMINAL = {"completed", "completed_with_errors", "failed", "cancelled"}
FIELD_LABEL = {
    "filename": "File name", "author": "Author", "document_type": "Document type", "language": "Language",
    "year": "Year", "year_gte": "Year", "year_lte": "Year", "years": "Year",
}
DOC_TYPES = ["research_paper", "book", "report", "documentation", "article", "other"]
DOC_TYPE_LABEL = {
    "research_paper": "Research paper", "book": "Book", "report": "Report",
    "documentation": "Documentation", "article": "Article", "other": "Other",
}
OCR_LANG_LABEL = {"eng": "English", "fas": "Persian", "ara": "Arabic", "deu": "German", "fra": "French", "rus": "Russian", "tur": "Turkish"}

# Colour policy: never set a text colour without also setting its own background (and vice versa),
# so nothing becomes unreadable in Streamlit's light OR dark theme. Everything else inherits the theme.
# Each pair is (background, foreground) and must keep a WCAG AA contrast of at least 4.5:1 (see tests).
PILL_COLORS = {
    "ok": ("#dcfce7", "#14532d"),
    "bad": ("#fee2e2", "#7f1d1d"),
    "warn": ("#fef3c7", "#78350f"),
    "unk": ("#e2e8f0", "#1e293b"),
}
BUTTON_COLORS = {"normal": ("#0f766e", "#ffffff"), "hover": ("#115e59", "#ffffff")}
_NEUTRAL_BORDER = "rgba(128,128,128,.35)"  # visible on both light and dark backgrounds
_NEUTRAL_FILL = "rgba(128,128,128,.10)"


def _build_css() -> str:
    pills = "\n".join(
        f".sd-pill.sd-{k} {{ background:{bg}; color:{fg}; }}" for k, (bg, fg) in PILL_COLORS.items()
    )
    (nb, nf), (hb, hf) = BUTTON_COLORS["normal"], BUTTON_COLORS["hover"]
    return f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1100px; }}
/* Keep mixed-language content (e.g. Persian answers) readable */
[data-testid="stChatMessage"] p, input, textarea {{ unicode-bidi: plaintext; text-align: start; }}
[data-testid="stMetric"] {{ background:{_NEUTRAL_FILL}; border:1px solid {_NEUTRAL_BORDER}; border-radius:14px; padding:14px 16px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:14px; border-color:{_NEUTRAL_BORDER}; }}
.stButton > button {{ border-radius:10px; }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background:{nb}; border-color:{nb}; color:{nf}; }}
.stButton > button[kind="primary"]:hover:not(:disabled), [data-testid="stBaseButton-primary"]:hover:not(:disabled) {{ background:{hb}; border-color:{hb}; color:{hf}; }}
.sd-hero {{ padding:6px 0 18px 0; border-bottom:1px solid {_NEUTRAL_BORDER}; margin-bottom:18px; }}
.sd-hero h1 {{ margin:0; font-size:1.8rem; }}
.sd-hero p {{ margin:4px 0 0 0; opacity:.8; }}
.sd-pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.85rem; font-weight:600; }}
{pills}
@media (max-width: 640px) {{ .block-container {{ padding: 1rem .8rem; }} .sd-hero h1 {{ font-size:1.4rem; }} }}
</style>
"""


CSS = _build_css()


# ---------------------------------------------------------------- backend auto-start
# `streamlit run ui/app.py` is enough: when the API address is local and nothing answers there,
# the UI starts the FastAPI backend itself (one uvicorn worker, log in api.log) and stops it on exit.
# Disable with SMARTDOCS_AUTOSTART_API=0 (e.g. when the API runs elsewhere).
def _is_local(url: str) -> bool:
    return (urlparse(url).hostname or "") in ("localhost", "127.0.0.1", "::1")


def start_backend(base_url: str, *, root: Path = PROJECT_ROOT, log_path: Path = API_LOG, env: dict | None = None) -> subprocess.Popen:
    port = urlparse(base_url).port or 8000
    log = open(log_path, "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
        cwd=str(root), stdout=log, stderr=subprocess.STDOUT, env=env,
    )
    atexit.register(lambda: proc.terminate() if proc.poll() is None else None)
    return proc


def wait_healthy(client: ApiClient, proc: subprocess.Popen | None, timeout: float = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.health()
            return True
        except ApiError:
            pass
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def read_log_tail(path: Path = API_LOG, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


@st.cache_resource(show_spinner="Starting the backend (the first start can take up to a minute)...")
def _ensure_backend(url: str) -> dict:
    client = ApiClient(url)
    try:
        client.health()
        return {"ok": True, "proc": None, "gave_up": False}
    except ApiError:
        pass
    if os.getenv("SMARTDOCS_AUTOSTART_API", "1") == "0" or not _is_local(url):
        return {"ok": False, "proc": None, "gave_up": False}
    proc = start_backend(url)
    return {"ok": wait_healthy(client, proc), "proc": proc, "gave_up": True}


def retry_backend() -> None:
    """Forget the cached start-up result so the next run tries to start the backend again."""
    _ensure_backend.clear()


def get_client() -> ApiClient:
    client = ApiClient()
    state = _ensure_backend(client.base_url)
    proc = state["proc"]
    if (proc is not None and proc.poll() is not None) or (not state["ok"] and not state["gave_up"]):
        _ensure_backend.clear()  # backend died, or an external one is still down: look again next run
    return client


def setup_page(title: str, subtitle: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="sd-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def nav_link(page: str, label: str, icon: str) -> None:
    """Page link that degrades gracefully when a page file is run outside st.navigation."""
    try:
        st.page_link(page, label=label, icon=icon, use_container_width=True)
    except Exception:
        st.caption(label)


def pill(text: str, kind: str = "unk") -> str:
    return f'<span class="sd-pill sd-{kind}">{text}</span>'


def error_text(err: ApiError) -> str:
    base = ERROR_TEXT.get(err.code) or err.message
    if err.code == "VALIDATION_ERROR" and err.details:
        base += " (" + "; ".join(f"{'.'.join(d['loc'][1:])}: {d['message']}" for d in err.details[:3]) + ")"
    return base


def show_error(err: ApiError, *, hint_code: bool = True) -> None:
    st.error(error_text(err))
    if err.code in ("API_UNREACHABLE", "API_TIMEOUT") and isinstance(err.details, dict):
        st.caption(f"Tried to reach: {err.details.get('url')}")
    if hint_code and err.request_id:
        st.caption(f"Error code: {err.code} · Request ID: {err.request_id}")


def safe_call(fn, *args, **kwargs):
    """Run an API call; on failure show an error and return None."""
    try:
        return fn(*args, **kwargs)
    except ApiError as err:
        show_error(err)
        return None


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def describe_filter(key: str, value) -> str:
    if key == "year_gte": return f">= {value}"
    if key == "year_lte": return f"<= {value}"
    if key == "years": return ", ".join(str(v) for v in value)
    if key == "document_type": return DOC_TYPE_LABEL.get(value, str(value))
    return str(value)


def unreachable_panel(client) -> None:
    """Shown when the API cannot be reached: explains the fix and retries automatically."""
    with st.expander("How to fix this", expanded=True):
        st.markdown(
            f"""
The UI is calling `{client.base_url}`.

1. **Local run:** the UI normally starts the backend by itself. If it could not, start it manually in a second terminal:
   `uvicorn src.api.main:app --workers 1`
2. **Backend elsewhere?** Set the `SMARTDOCS_API_URL` environment variable to its address (and `SMARTDOCS_AUTOSTART_API=0`).
3. **Port in use?** Another program may occupy that port; stop it or change the port in `SMARTDOCS_API_URL`.
"""
        )
        if st.button("Try to start the backend again"):
            retry_backend()
            st.rerun()
        tail = read_log_tail()
        if tail:
            st.caption("Last lines of the backend log (api.log):")
            st.code(tail, language="text")

    @st.fragment(run_every=4)
    def _auto_retry() -> None:
        try:
            client.health()
        except ApiError:
            st.caption("Waiting for the backend... (retrying automatically)")
            return
        st.rerun(scope="app")

    _auto_retry()
