import streamlit as st

from api_client import ApiError
from common import (
    DOC_TYPES, DOC_TYPE_LABEL, FIELD_LABEL, TRANSIENT, describe_filter, error_text, get_client, nav_link, setup_page, show_error,
)

setup_page("Chat with documents", "Answers are built only from passages actually retrieved from your documents")
client = get_client()
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("pending_question", None)

try:
    stats = client.stats()
except ApiError as err:
    show_error(err)
    st.stop()
if stats.get("chunk_count") == 0:
    st.info(error_text(ApiError("INDEX_EMPTY", "")))
    nav_link("views/upload.py", "Go to Upload documents", ":material/upload_file:")
    st.stop()

MF = ["mf_filename", "mf_type", "mf_lang", "mf_author", "mf_gte", "mf_lte"]
ss.setdefault("mf_filename", "—"); ss.setdefault("mf_type", "—"); ss.setdefault("mf_lang", "")
ss.setdefault("mf_author", ""); ss.setdefault("mf_gte", 0); ss.setdefault("mf_lte", 0)


def manual_filter() -> dict:
    out = {}
    if ss.mf_filename != "—": out["filename"] = ss.mf_filename
    if ss.mf_type != "—": out["document_type"] = ss.mf_type
    if ss.mf_lang.strip(): out["language"] = ss.mf_lang.strip().lower()
    if ss.mf_author.strip(): out["author"] = ss.mf_author.strip()
    if ss.mf_gte: out["year_gte"] = int(ss.mf_gte)
    if ss.mf_lte: out["year_lte"] = int(ss.mf_lte)
    return out


def load_into_manual(applied: dict) -> None:
    ss.mf_filename = applied.get("filename", "—")
    ss.mf_type = applied.get("document_type", "—")
    ss.mf_lang = applied.get("language", "")
    ss.mf_author = applied.get("author", "")
    ss.mf_gte = applied.get("year_gte", applied.get("year", 0))
    ss.mf_lte = applied.get("year_lte", applied.get("year", 0))


with st.expander("Search settings and manual filter", expanded=False):
    a, b = st.columns(2)
    auto_filter = a.toggle("Extract filters from the question (LLM)", value=True)
    k = b.slider("Passages to retrieve", 1, 10, 4)
    docs = []
    try:
        docs = sorted({d["filename"] for d in client.documents()["documents"] if d["filename"]})
    except ApiError:
        pass
    c1, c2, c3 = st.columns(3)
    c1.selectbox("Only in this file", ["—"] + docs, key="mf_filename")
    c2.selectbox("Document type", ["—"] + DOC_TYPES, key="mf_type", format_func=lambda v: DOC_TYPE_LABEL.get(v, v))
    c3.text_input("Language (2-letter code, e.g. en)", key="mf_lang", max_chars=5)
    d1, d2, d3 = st.columns(3)
    d1.text_input("Author", key="mf_author")
    d2.number_input("Year from (0 = no limit)", 0, 2100, key="mf_gte", step=1)
    d3.number_input("Year to (0 = no limit)", 0, 2100, key="mf_lte", step=1)
    st.caption("The manual filter is combined with the extracted one; if both set the same field, the manual value wins.")


def render_filters(meta: dict, *, latest: bool) -> None:
    status = meta["filter_status"]
    with st.expander("Detected filters", expanded=bool(meta["applied_filter"]) or status == "failed"):
        if status == "failed":
            st.warning("Automatic filter extraction failed; the search ran without an automatic filter. " + (meta.get("filter_error") or ""))
        elif status == "skipped":
            st.caption("Automatic filter extraction was turned off.")
        elif status == "none_detected" and not meta["manual_filter"]:
            st.caption("No metadata constraint was found in the question; a normal search was run.")
        rows = []
        for key, value in meta["applied_filter"].items():
            manual = key in meta["manual_filter"]
            rows.append({"Field": FIELD_LABEL.get(key, key), "Condition / value": describe_filter(key, value),
                         "Source": "Manual" if manual else "Automatic (LLM)", "Status": "Applied to the search"})
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        if latest and (meta["applied_filter"]):
            st.button("Edit filters (copy to manual filter)", on_click=load_into_manual,
                      args=(meta["applied_filter"],), key=f"edit_{len(ss.messages)}")


def render_sources(meta: dict) -> None:
    if not meta["sources"]:
        return
    st.markdown("**Retrieved sources**")
    for s in meta["sources"]:
        title = f"[{s['index']}] {s['filename'] or 'unknown'}" + (f" - page {s['page']}" if s["page"] is not None else "")
        with st.expander(f"{title} - similarity {s['similarity']:.2f}"):
            st.write(s["content"])


def render_message(msg: dict, idx: int) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
            return
        st.write(msg["content"])
        meta = msg.get("meta")
        if meta:
            if meta["status"] == "no_evidence":
                st.warning("No sufficient evidence was found in the documents; no answer was generated.")
            render_filters(meta, latest=idx == len(ss.messages) - 1)
            render_sources(meta)
            t = meta["timings_ms"]
            st.caption("Time (ms): " + " · ".join(f"{n} {v}" for n, v in t.items()) + f" · model: {meta['model']}")


def ask(question: str) -> None:
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages if not m.get("error")]
    ss.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Extracting filters, searching and generating the answer..."):
            res = client.chat(question, history=history, k=k, auto_filter=auto_filter, manual_filter=manual_filter() or None)
        ss.messages.append({"role": "assistant", "content": res["answer"], "meta": res})
    except ApiError as err:
        ss.messages.append({"role": "assistant", "content": error_text(err), "error": True, "transient": err.code in TRANSIENT, "question": question})


top = st.columns([1, 1, 4])
if top[0].button("Clear history", disabled=not ss.messages):
    ss.messages = []
    st.rerun()
last_user = next((m["content"] for m in reversed(ss.messages) if m["role"] == "user"), None)
if top[1].button("Resend last question with current filters", disabled=last_user is None):
    ss.pending_question = last_user

for i, m in enumerate(ss.messages):
    render_message(m, i)
if ss.messages and ss.messages[-1].get("error") and ss.messages[-1].get("transient"):
    if st.button("Retry"):
        ss.pending_question = ss.messages[-1]["question"]
        ss.messages = ss.messages[:-2]
        st.rerun()

prompt = st.chat_input("Ask a question about your documents...")
question = prompt or ss.pending_question
if question:
    ss.pending_question = None
    ask(question)
    st.rerun()
