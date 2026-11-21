import hashlib
import json
import time

import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

if __package__:
    from .rag import (
        EMBEDDING_MODEL,
        build_index,
        generate_answer,
        retrieve,
    )
else:
    from rag import (
        EMBEDDING_MODEL,
        build_index,
        generate_answer,
        retrieve,
    )


st.set_page_config(
    page_title="LectureLens",
    page_icon="📚",
    layout="wide",
)

st.title("📚 LectureLens")
st.write("Ask questions about your lecture PDFs and inspect the evidence.")
st.caption(
    "Text-based PDFs • Local models • Filename and PDF page references"
)


@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def document_signature(files):
    digest = hashlib.sha256()

    for filename, content in files:
        digest.update(filename.encode("utf-8"))
        digest.update(hashlib.sha256(content).digest())

    return digest.hexdigest()


def show_evidence(hits):
    if not hits:
        return

    with st.expander("Retrieved passages — verify the answer"):
        for number, hit in enumerate(hits, start=1):
            st.text(
                f"[S{number}] {hit['source']} | "
                f"PDF page {hit['page']} | "
                f"similarity {hit['score']:.3f}"
            )
            st.text(hit["text"])
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Your lecture notes")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    model_name = st.selectbox(
        "Answer model",
        ["llama3.2:3b", "llama3.2:1b"],
    )

    top_k = st.slider("Maximum retrieved passages", 1, 8, 5)

    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
    )

    st.caption(
        "Similarity is a retrieval score, not answer confidence. "
        "Tune the threshold using evaluation questions."
    )

    files = [
        (uploaded.name, uploaded.getvalue())
        for uploaded in (uploaded_files or [])
    ]
    signature = document_signature(files)

    if st.button("Build / rebuild knowledge base", type="primary"):
        if not files:
            st.warning("Upload at least one PDF first.")
        else:
            try:
                with st.spinner("Reading PDFs and creating embeddings..."):
                    embedder = load_embedder()
                    index, warnings = build_index(files, embedder)

                st.session_state.index = index
                st.session_state.signature = signature
                st.session_state.messages = []
                st.session_state.extraction_warnings = warnings
                st.success("Knowledge base ready.")

            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    if st.button("Clear conversation"):
        st.session_state.messages = []

    if "index" in st.session_state:
        st.metric(
            "Indexed passages",
            len(st.session_state.index["chunks"]),
        )

    for warning in st.session_state.get("extraction_warnings", []):
        st.warning(warning)

    if st.session_state.messages:
        st.download_button(
            "Download conversation",
            data=json.dumps(
                st.session_state.messages,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="lecturelens-conversation.json",
            mime="application/json",
        )


ready = (
    bool(files)
    and "index" in st.session_state
    and st.session_state.get("signature") == signature
)

if not ready:
    st.info(
        "Upload PDFs and build the knowledge base. "
        "Rebuild whenever the selected documents change."
    )

st.caption(
    "Ask complete questions. This version does not use previous messages "
    "to interpret follow-up questions. Page references use PDF page order."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "seconds" in message:
            st.caption(f"Response time: {message['seconds']:.1f} seconds")

        show_evidence(message.get("hits", []))


question = st.chat_input(
    "Example: What is the difference between classification and regression?",
    disabled=not ready,
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            started = time.perf_counter()

            with st.spinner("Retrieving passages and preparing an answer..."):
                hits = retrieve(
                    question,
                    st.session_state.index,
                    load_embedder(),
                    top_k=top_k,
                    min_score=min_score,
                )
                answer = generate_answer(question, hits, model_name)

            elapsed = time.perf_counter() - started
            st.markdown(answer)
            st.caption(f"Response time: {elapsed:.1f} seconds")
            show_evidence(hits)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "hits": hits,
                    "seconds": elapsed,
                }
            )

        except requests.RequestException as exc:
            st.error(
                "Ollama request failed. Check that Ollama is running "
                f"and the selected model is downloaded. Details: {exc}"
            )

        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")