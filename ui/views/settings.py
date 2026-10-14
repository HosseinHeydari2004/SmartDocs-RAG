import streamlit as st

from api_client import ApiError
from common import get_client, pill, safe_call, setup_page, show_error

setup_page("Settings", "OpenRouter connection, OCR status and embedding configuration")
client = get_client()

status = safe_call(client.settings_status)
if status is None:
    st.stop()
llm, ocr = status["llm"], status["ocr"]

with st.container(border=True):
    st.subheader("OpenRouter")
    source = {"env": "environment variable", "session": "current server session", "none": "-"}[llm["key_source"]]
    st.markdown(
        pill("API key configured" if llm["key_configured"] else "API key missing", "ok" if llm["key_configured"] else "bad")
        + f" &nbsp; Key source: {source} &nbsp; | &nbsp; Current model: <code>{llm['model']}</code>",
        unsafe_allow_html=True,
    )
    with st.form("llm_form", clear_on_submit=True):
        key = st.text_input("API key", type="password", placeholder="sk-or-...", help="The key is never displayed or stored on disk.")
        model = st.text_input("Model name", value=llm["model"], help="For example openrouter/auto or any OpenRouter model id")
        submitted = st.form_submit_button("Save settings", type="primary")
    if submitted:
        res = safe_call(client.set_llm, key.strip() or None, model.strip() or None)
        if res:
            st.success("Settings applied in server memory.")
            st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("Test connection and key"):
        with st.spinner("Sending one tiny test request..."):
            try:
                client.test_llm()
                st.success("Connection works and the key is valid.")
            except ApiError as err:
                show_error(err)
    if llm["key_source"] == "session" and c2.button("Forget session key"):
        safe_call(client.clear_key)
        st.rerun()
    st.caption(
        "The key is kept **in the backend process memory only**: it is lost on restart, never appears in logs or API responses, "
        "and the browser does not keep it after submission. For persistence put OPENROUTER_API_KEY in a .env file (kept out of Git). "
        "Use HTTPS in any real deployment."
    )

with st.container(border=True):
    st.subheader("OCR")
    st.markdown(pill("Ready" if ocr["available"] else "Unavailable", "ok" if ocr["available"] else "warn"), unsafe_allow_html=True)
    if ocr["available"]:
        st.write(f"Engine: Tesseract {ocr['version']} | Installed languages: {', '.join(ocr['languages'])}")
        st.write("Scanned PDF support (Poppler):", "yes" if ocr["pdf_supported"] else "no")
    else:
        st.warning(ocr.get("reason") or "Tesseract is not available.")
        st.markdown(
            "**Fix:** install Tesseract, then either add its folder to `PATH` or put its full path in `.env`:\n\n"
            "`TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe`\n\n"
            "Restart the app afterwards (a terminal opened before installing does not see the new PATH)."
        )
    if ocr["available"] and not ocr["pdf_supported"]:
        st.info("Scanned **PDF** files also need Poppler. Install it and set `POPPLER_PATH` to its `bin` folder (e.g. `C:\\poppler\\Library\\bin`).")

with st.container(border=True):
    st.subheader("Embedding and limits")
    emb, lim = status["embedding"], status["limits"]
    st.write("Model:", f"`{emb['model']}`", "| device:", f"`{emb['device_setting']}`", "| batch size:", emb["batch_size"])
    st.caption("These values come from the backend environment variables. Changing the embedding model requires clearing the collection and re-indexing.")
    st.write(f"Max file size: {lim['upload_max_mb']} MB | Max files per request: {lim['upload_max_files']} | OCR page limit: {lim['ocr_max_pages']}")
