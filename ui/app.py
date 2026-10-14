"""SmartDocs-RAG - Streamlit front-end. Run: ``streamlit run ui/app.py``."""
import streamlit as st

st.set_page_config(page_title="SmartDocs-RAG", page_icon=":material/description:", layout="wide")

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page("views/upload.py", title="Upload documents", icon=":material/upload_file:"),
    st.Page("views/chat.py", title="Chat with documents", icon=":material/chat:"),
    st.Page("views/documents.py", title="Manage documents", icon=":material/folder_open:"),
    st.Page("views/settings.py", title="Settings", icon=":material/settings:"),
]
st.navigation(pages).run()
