"""
app.py
------
Streamlit chat interface for the RAG chatbot.

Responsibilities:
  - keep the multi-turn conversation in session state,
  - send each new question (plus history) to the RAG pipeline,
  - display the answer AND the page/source citations beneath it.
"""

import streamlit as st

from rag import RAGPipeline

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with the Document")
st.caption("Answers come only from the ingested PDF. Sources are shown below each reply.")


# Build the pipeline once and cache it, so we don't reconnect to Qdrant /
# rebuild the model on every keystroke. @st.cache_resource is the right tool
# for objects that should live for the whole session.
@st.cache_resource
def get_pipeline():
    return RAGPipeline()


# Surface configuration errors (e.g. missing API keys) clearly instead of a
# blank screen, so the cause is obvious.
try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

# Conversation history lives in session state: a list of message dicts.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render the whole conversation each run (Streamlit reruns top-to-bottom).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            refs = ", ".join(f"{src} (page {pg})" for src, pg in msg["sources"])
            st.caption(f"📑 Sources: {refs}")

# Chat input box pinned to the bottom of the page.
if prompt := st.chat_input("Ask a question about the document..."):
    # Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass prior turns so follow-up questions ("what about that?") work.
    history = [(m["role"], m["content"]) for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            result = pipeline.answer(prompt, history=history)
        st.markdown(result["answer"])
        if result["sources"]:
            refs = ", ".join(f"{src} (page {pg})" for src, pg in result["sources"])
            st.caption(f"📑 Sources: {refs}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
