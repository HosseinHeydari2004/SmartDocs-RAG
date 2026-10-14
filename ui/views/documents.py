import streamlit as st

from common import DOC_TYPE_LABEL, get_client, nav_link, safe_call, setup_page

setup_page("Manage documents", "Indexed documents, their metadata and controlled deletion")
client = get_client()

data = safe_call(client.documents)
if data is None:
    st.stop()
docs = data["documents"]
a, b = st.columns(2)
a.metric("Documents", data["total_documents"])
b.metric("Chunks", data["total_chunks"])

if not docs:
    st.info("No documents have been indexed yet.")
    nav_link("views/upload.py", "Upload documents", ":material/upload_file:")
else:
    st.dataframe(
        [{"File": d["filename"], "Title": d["title"], "Author": d["author"], "Year": d["year"],
          "Type": DOC_TYPE_LABEL.get(d["document_type"], d["document_type"]), "Language": d["language"],
          "Extraction": {"scanned": "OCR", "digital": "Text"}.get(d["kind"], d["kind"]), "Chunks": d["chunk_count"]} for d in docs],
        hide_index=True, use_container_width=True,
    )
    labels = {f"{d['filename']} ({(d['doc_id'] or 'legacy')[:8]})": d for d in docs}
    choice = st.selectbox("Select a document to inspect or delete", list(labels))
    doc = labels[choice]
    st.caption("Document metadata")
    st.json({k: v for k, v in doc.items() if v is not None})
    if doc["doc_id"] is None:
        st.warning("This document was indexed by an older version and has no id, so it cannot be deleted individually. Clear the collection and re-index to remove it.")
    else:
        confirm = st.checkbox(f"I confirm deleting '{doc['filename']}'", key=f"confirm_del_{doc['doc_id']}")
        if st.button("Delete this document", disabled=not confirm, type="primary"):
            res = safe_call(client.delete_document, doc["doc_id"])
            if res:
                st.success(f"Deleted {res['deleted_chunks']} chunk(s).")
                st.rerun()

with st.expander("Clear the whole collection (irreversible)"):
    st.error("All documents and chunks of this collection will be deleted. Other collections and files are untouched.")
    c1 = st.checkbox("I understand this cannot be undone")
    c2 = st.text_input("Type DELETE to confirm")
    if st.button("Clear collection", disabled=not (c1 and c2.strip() == "DELETE")):
        res = safe_call(client.clear_documents)
        if res:
            st.success(f"Deleted {res['deleted_documents']} document(s) and {res['deleted_chunks']} chunk(s).")
            st.rerun()
