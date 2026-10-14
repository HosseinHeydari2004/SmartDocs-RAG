import streamlit as st

from api_client import ApiError
from common import get_client, nav_link, pill, setup_page, show_error, unreachable_panel

setup_page("SmartDocs-RAG", "Ask questions about your documents, with automatic metadata filtering and real source citations")
client = get_client()

try:
    client.health()
    api_ok = True
except ApiError as err:
    api_ok = False
    show_error(err)
    if err.code in ("API_UNREACHABLE", "API_TIMEOUT"):
        unreachable_panel(client)

stats = settings = None
if api_ok:
    try:
        stats = client.stats()
        settings = client.settings_status()
    except ApiError as err:
        show_error(err)

unknown = "Unknown"
c1, c2, c3, c4 = st.columns(4)
c1.metric("FastAPI service", "Connected" if api_ok else "Offline")
c2.metric("Chroma database", {"ok": "Healthy", "error": "Error"}.get(stats["chroma_status"], unknown) if stats else unknown)
c3.metric("Indexed documents", stats["document_count"] if stats and stats.get("document_count") is not None else unknown)
c4.metric("Indexed chunks", stats["chunk_count"] if stats and stats.get("chunk_count") is not None else unknown)

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("LLM service")
    if settings:
        llm = settings["llm"]
        st.markdown(pill("API key configured" if llm["key_configured"] else "API key missing", "ok" if llm["key_configured"] else "bad"), unsafe_allow_html=True)
        st.write("Provider: OpenRouter")
        st.write("Model:", f"`{llm['model']}`")
    else:
        st.write(unknown)
with right, st.container(border=True):
    st.subheader("Embedding and OCR")
    if stats and settings:
        st.write("Embedding model:", f"`{stats['embedding_model']}`")
        st.write("Model loaded:", "yes" if stats["embedding_loaded"] else "not yet (loads on first use)")
        ocr = settings["ocr"]
        st.markdown("OCR: " + pill("Ready" if ocr["available"] else "Unavailable", "ok" if ocr["available"] else "warn"), unsafe_allow_html=True)
        if stats.get("embedding_mismatch"):
            st.warning("The configured embedding model differs from the one used for the existing index. Re-index to continue.")
    else:
        st.write(unknown)

st.subheader("Quick access")
q1, q2, q3 = st.columns(3)
with q1: nav_link("views/upload.py", "Upload documents", ":material/upload_file:")
with q2: nav_link("views/chat.py", "Chat with documents", ":material/chat:")
with q3: nav_link("views/settings.py", "Settings", ":material/settings:")
