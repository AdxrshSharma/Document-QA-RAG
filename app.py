import time
from pathlib import Path

import streamlit as st

from rag_core import (DATA_DIR, OLLAMA_MODEL, ask, ask_llm, ingest, list_indexed_docs,
                      reset_index, warm_up)

st.set_page_config(page_title="Local Document Q&A")
st.title("Local Document Q&A (100% free)")
st.caption("Upload your own PDFs, then ask questions. Everything runs on your computer.")


@st.cache_resource(show_spinner="Loading models (only once per session)...")
def load_models():
    warm_up()
    return True


load_models()
data_dir = Path(DATA_DIR)
data_dir.mkdir(exist_ok=True)

# ---------------- Sidebar: upload + index ----------------
with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    replace = st.checkbox("Remove previously indexed documents first", value=True)
    if st.button("Index uploaded PDFs", disabled=not uploads):
        try:
            with st.spinner("Reading and indexing..."):
                if replace:
                    reset_index()
                    for old in data_dir.glob("*.pdf"):
                        old.unlink()
                for f in uploads:
                    (data_dir / Path(f.name).name).write_bytes(f.getbuffer())
                n_chunks = ingest()
            indexed = set(list_indexed_docs())
            st.success(f"Indexed {n_chunks} chunks from {len(indexed)} document(s).")
            missing = [f.name for f in uploads if Path(f.name).name not in indexed]
            if missing:
                st.warning("No readable text found in: " + ", ".join(missing) +
                           ". These may be scanned images (this project needs text-based PDFs).")
        except SystemExit as e:
            st.error(str(e))

    st.header("Answer engine")
    engine = st.radio("How should answers be written?",
                      ["Extractive (no install)", "Local LLM via Ollama (better)"],
                      help="Extractive copies an exact phrase from the document: good for facts like "
                           "numbers and names. The Ollama LLM can also handle lists and summaries.")
    model_name = OLLAMA_MODEL
    if engine.startswith("Local LLM"):
        model_name = st.text_input("Ollama model", value=OLLAMA_MODEL)

    docs = list_indexed_docs()
    st.subheader("Currently indexed")
    if docs:
        for d in docs:
            st.write(f"- {d}")
    else:
        st.write("Nothing yet.")

# ---------------- Main: ask questions ----------------
if not docs:
    st.info("Upload one or more PDFs in the sidebar and click **Index uploaded PDFs**.")
    st.stop()

q = st.text_input("Ask a question about your documents")
if q:
    t0 = time.time()
    if engine.startswith("Local LLM"):
        try:
            with st.spinner("Searching and writing an answer (a local LLM can take a while on CPU)..."):
                text, chunks = ask_llm(q, model=model_name)
            st.success(text)
            st.caption(f"answered in {time.time() - t0:.1f}s")
        except RuntimeError as e:
            st.error(str(e))
            chunks = []
    else:
        with st.spinner("Searching and reading..."):
            best, chunks = ask(q)
        if best["source"] is None:
            st.warning(best["answer"])
        else:
            st.success(best["answer"])
            st.caption(f"Source: {best['source']} p.{best['page']}  |  confidence {best['score']:.0%}  |  "
                       f"answered in {time.time() - t0:.1f}s")
    with st.expander("Retrieved passages"):
        for c in chunks:
            st.caption(f"{c['source']} p.{c['page']} (distance {c['distance']:.3f})")
            st.write(c["text"])