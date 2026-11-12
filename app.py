"""Streamlit UI.  Run with:  streamlit run app.py

Set DEMO_MODE=1 for a public deployment: every visitor gets a private, temporary index
and a question limit, so strangers can't see each other's PDFs or burn your API credits.
"""
import os
import tempfile
from pathlib import Path

import anthropic
import streamlit as st

from docchat.embeddings import Embedder
from docchat.pipeline import DocChat, cited_hits
from docchat.store import VectorStore

DEMO_MODE = os.getenv("DEMO_MODE") == "1"
MAX_QUESTIONS = int(os.getenv("DEMO_MAX_QUESTIONS", "15"))

st.set_page_config(page_title="DocChat", page_icon="📄")
st.title("📄 DocChat")
st.caption("Ask questions about your PDFs. Every answer cites its sources.")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set. Add it to a .env file (or the host's secrets) and restart.")
    st.stop()

if DEMO_MODE:
    st.info(
        f"Demo mode: your PDFs are private to your session (not shared with other visitors), "
        f"and you can ask up to {MAX_QUESTIONS} questions. Please don't upload confidential files."
    )


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~130 MB)...")
def get_embedder() -> Embedder:
    embedder = Embedder()
    embedder.model  # force-load now so the first question isn't slow
    return embedder  # the model is shared by all sessions; the index is not


def get_chat() -> DocChat:
    if "chat" not in st.session_state:
        if DEMO_MODE:
            store = VectorStore(Path(tempfile.mkdtemp(prefix="docchat_")))  # private to this session
        else:
            store = VectorStore.load()  # persistent index in data/index
        st.session_state.chat = DocChat(embedder=get_embedder(), store=store)
    return st.session_state.chat


chat = get_chat()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "asked" not in st.session_state:
    st.session_state.asked = 0


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['doc']}, page {s['page']}** · similarity {s['score']:.2f}")
            st.caption(s["text"])


# ---------- Sidebar: upload + manage documents ----------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index documents", disabled=not files, width="stretch"):
        for f in files:
            try:
                with st.spinner(f"Indexing {f.name}..."):
                    added = chat.ingest(f.getvalue(), f.name)
                if added:
                    st.success(f"{f.name}: {added} chunks indexed")
                else:
                    st.info(f"{f.name}: already indexed")
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:  # corrupted / encrypted PDFs etc.
                st.error(f"{f.name}: could not be read ({e})")

    docs = chat.store.documents()
    if docs:
        st.subheader("Indexed")
        for name, n in docs.items():
            st.write(f"• {name} ({n} chunks)")
        if st.button("Clear all documents", width="stretch"):
            chat.store.reset()
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No documents yet. Upload a PDF to begin.")

# ---------- Chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources", []))

if question := st.chat_input("Ask something about your documents"):
    if DEMO_MODE and st.session_state.asked >= MAX_QUESTIONS:
        st.warning(
            f"Demo limit reached ({MAX_QUESTIONS} questions per session). "
            "Clone the GitHub repo to run it locally without limits."
        )
        st.stop()
    st.session_state.asked += 1

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not chat.store.documents():
            answer, sources = "Please upload and index a PDF first.", []
            st.markdown(answer)
        else:
            try:
                hits, stream = chat.ask(question)
                answer = st.write_stream(stream)
                sources = [
                    {"n": n, "doc": h.chunk.doc, "page": h.chunk.page, "score": h.score, "text": h.chunk.text}
                    for n, h in cited_hits(answer, hits)
                ]
                show_sources(sources)
            except anthropic.APIError as e:
                answer, sources = f"Claude API error: {e}", []
                st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
