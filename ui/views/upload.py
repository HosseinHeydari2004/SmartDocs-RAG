import streamlit as st

from api_client import ApiError
from common import (
    ERROR_TEXT, FILE_STATUS, JOB_STATUS, STAGE_TEXT, TERMINAL, OCR_LANG_LABEL,
    get_client, human_size, setup_page, show_error, safe_call,
)

setup_page("Upload documents", "Index documents in batches; progress is read from the real pipeline state")
client = get_client()
ss = st.session_state
for key, default in (("active_job", None), ("last_job", None), ("upload_nonce", 0), ("retry_payload", None)):
    ss.setdefault(key, default)

settings = safe_call(client.settings_status)
if settings is None:
    st.stop()
limits, ocr = settings["limits"], settings["ocr"]
max_bytes = limits["upload_max_mb"] * 1024 * 1024
busy = ss.active_job is not None


def start_job(scanned: bool, files: list[tuple[str, bytes, str]], on_duplicate: str, ocr_lang: str) -> None:
    try:
        job = client.upload(scanned, files, on_duplicate, ocr_lang)
    except ApiError as err:
        show_error(err)
        return
    ss.active_job, ss.last_job = job["id"], None
    ss.upload_nonce += 1
    ss.retry_payload = {"scanned": scanned, "files": {f[0]: f for f in files}, "ocr_lang": ocr_lang}
    st.rerun()


def selection_table(uploaded) -> bool:
    rows, ok = [], True
    for f in uploaded:
        too_big = f.size > max_bytes
        ok = ok and not too_big
        rows.append({"File": f.name, "Size": human_size(f.size),
                     "Format": f.name.rsplit(".", 1)[-1].upper() if "." in f.name else "-",
                     "Status": "Over size limit" if too_big else "Ready"})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if not ok:
        st.error(f"Each file may be at most {limits['upload_max_mb']} MB.")
    if len(uploaded) > limits["upload_max_files"]:
        st.error(f"At most {limits['upload_max_files']} files per request.")
        ok = False
    return ok


def upload_panel(scanned: bool) -> None:
    exts = limits["scanned_extensions"] if scanned else limits["digital_extensions"]
    if scanned and not ocr["pdf_supported"]:
        exts = [e for e in exts if e != ".pdf"]
    uploaded = st.file_uploader(
        "Select files", accept_multiple_files=True, type=[e.lstrip(".") for e in exts],
        key=f"{'scan' if scanned else 'digital'}_{ss.upload_nonce}", disabled=busy,
        help=f"Allowed: {', '.join(exts)} - max {limits['upload_max_mb']} MB per file",
    )
    ocr_lang = "eng"
    if scanned:
        langs = ocr["languages"]
        default = langs.index("eng") if "eng" in langs else 0
        ocr_lang = st.selectbox("Document language (OCR)", langs, index=default,
                                format_func=lambda c: f"{OCR_LANG_LABEL.get(c, c)} ({c})", disabled=busy)
        if "fas" not in langs:
            st.caption("The Persian language pack (fas) is not installed in Tesseract; install it for Persian OCR.")
    policy = st.radio("If the same file is already indexed", ["skip", "replace"], horizontal=True, disabled=busy,
                      format_func=lambda v: "Skip it" if v == "skip" else "Replace the old version",
                      key=f"policy_{scanned}")
    ok = bool(uploaded) and selection_table(uploaded)
    if st.button("Start processing", type="primary", disabled=busy or not ok, key=f"go_{scanned}"):
        start_job(scanned, [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploaded], policy, ocr_lang)


tab_digital, tab_scanned = st.tabs(["Digital documents", "Scanned documents"])
with tab_digital:
    st.write("PDF with selectable text, DOCX, TXT, Markdown and CSV. Text is read directly from the file.")
    upload_panel(False)
with tab_scanned:
    st.write("Scanned PDFs and images (PNG, JPG, TIFF, BMP, WEBP). Text is extracted from page images with **OCR**.")
    st.info("OCR reads text only: tables, formulas and charts are not interpreted, and accuracy depends on scan quality.")
    if not ocr["available"]:
        st.warning("Scanned uploads are disabled: " + (ocr.get("reason") or "the OCR engine (Tesseract) is not available."))
        st.caption("See Settings > OCR for the fix.")
    else:
        if not ocr["pdf_supported"]:
            st.caption("Poppler was not found (set POPPLER_PATH or add it to PATH): images are supported, scanned PDFs are not.")
        upload_panel(True)


def file_rows(job: dict) -> list[dict]:
    rows = []
    for f in job["files"]:
        note = ERROR_TEXT.get(f["error_code"], f["error"]) if f["error_code"] else ""
        if f["warnings"]:
            note = (note + " | " if note else "") + " | ".join(f["warnings"])
        rows.append({"File": f["filename"], "Status": FILE_STATUS.get(f["status"], f["status"]),
                     "Chunks": f["chunks"], "Details": note})
    return rows


@st.fragment(run_every=1)
def live_monitor() -> None:
    job_id = ss.active_job
    if not job_id:
        return
    try:
        job = client.job(job_id)
    except ApiError as err:
        show_error(err)
        if err.code == "JOB_NOT_FOUND":
            ss.active_job = None
            st.rerun(scope="app")
        return
    st.progress(min(job["percent"], 100) / 100, text=f"{STAGE_TEXT.get(job['stage'], job['stage'])} - {job['percent']}%")
    a, b, c, d = st.columns(4)
    a.metric("Status", JOB_STATUS.get(job["status"], job["status"]))
    b.metric("Succeeded", job["files_succeeded"])
    c.metric("Failed", job["files_failed"])
    d.metric("Duplicates", job["files_skipped"])
    st.dataframe(file_rows(job), hide_index=True, use_container_width=True)
    if job["status"] in TERMINAL:
        ss.last_job, ss.active_job = job, None
        st.rerun(scope="app")
    elif st.button("Cancel processing", key="cancel_job"):
        safe_call(client.cancel_job, job_id)


def finished_report(job: dict) -> None:
    st.subheader("Processing result")
    ok = job["status"] == "completed"
    msg = (f"{job['documents_stored']} document(s) and {job['chunks_stored']} chunk(s) were stored in Chroma"
           + (f" (index total: {job['index_total_chunks']} chunks)." if job["index_total_chunks"] is not None else "."))
    (st.success if ok else st.warning if job["status"] != "failed" else st.error)(f"{JOB_STATUS[job['status']]} - {msg}")
    st.dataframe(file_rows(job), hide_index=True, use_container_width=True)
    failed = [f["filename"] for f in job["files"] if f["status"] == "failed"]
    payload = ss.retry_payload
    c1, c2 = st.columns(2)
    if failed and payload and any(n in payload["files"] for n in failed):
        if c1.button("Retry failed files", disabled=busy):
            start_job(payload["scanned"], [payload["files"][n] for n in failed if n in payload["files"]], "skip", payload["ocr_lang"])
    if c2.button("Dismiss report"):
        ss.last_job = None
        st.rerun()


st.divider()
if ss.active_job:
    live_monitor()
elif ss.last_job:
    finished_report(ss.last_job)
