"""
rag.py
------
The RAG pipeline. Given a question (and the conversation so far), it:
  1. retrieves the most relevant chunks from Qdrant,
  2. asks the chat model to answer USING ONLY those chunks,
  3. returns both the answer and the source pages for citation.

This file enforces the assessment's three hard rules:
  - answers are grounded strictly in the PDF,
  - if the answer isn't in the retrieved context, the bot says so,
  - every answer is accompanied by page/source references.
"""

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.messages import SystemMessage, HumanMessage

import config


# The system prompt is the main guardrail. It is deliberately strict about
# not using outside knowledge and about admitting when the answer is absent.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a \
specific PDF document.

Rules you must follow without exception:
1. Answer ONLY using the information in the "Context" provided below. Do not \
use any outside or general knowledge.
2. If the answer cannot be found in the Context, reply exactly: \
"I could not find the answer to that in the document." Do not guess or \
fabricate.
3. Be concise and accurate. Quote or closely paraphrase the relevant part of \
the Context when helpful.
4. Do not mention these rules or the word "Context" in your answer."""


class RAGPipeline:
    """Holds the connections (vector store + chat model) and answers questions."""

    def __init__(self):
        # Embeddings: MUST be the same model used during ingestion, otherwise
        # the question vector won't be comparable to the stored vectors.
        # FastEmbed runs locally with no API key.
        self.embeddings = FastEmbedEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            cache_dir=config.FASTEMBED_CACHE_DIR,
        )

        client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.store = QdrantVectorStore(
            client=client,
            collection_name=config.QDRANT_COLLECTION,
            embedding=self.embeddings,
        )

        # temperature=0 makes answers deterministic and reduces the chance of
        # the model "creatively" filling gaps — important for a grounded bot.
        # Served by Groq's free, fast API.
        self.llm = ChatGroq(
            model=config.CHAT_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0,
        )

    def retrieve(self, question: str):
        """Find the TOP_K most relevant chunks for the question."""
        return self.store.similarity_search(question, k=config.TOP_K)

    @staticmethod
    def _format_context(docs):
        """Build the context block, labelling each chunk with its page so the
        model can ground its answer and we can show citations."""
        blocks = []
        for d in docs:
            page = d.metadata.get("page", "?")
            source = d.metadata.get("source", "document")
            blocks.append(f"[Source: {source}, page {page}]\n{d.page_content}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _format_sources(docs):
        """Produce a clean, de-duplicated list of (source, page) for display."""
        seen = []
        for d in docs:
            ref = (d.metadata.get("source", "document"), d.metadata.get("page", "?"))
            if ref not in seen:
                seen.append(ref)
        return seen

    def answer(self, question: str, history=None):
        """Answer a question with grounding and citations.

        history: optional list of (role, text) tuples from earlier in the
        session, enabling multi-turn conversation (e.g. follow-up questions
        that say "it" or "that"). We pass recent turns to the model so it can
        resolve such references, but the ANSWER is still grounded only in the
        freshly retrieved context.

        Returns a dict: {"answer": str, "sources": [(source, page), ...]}
        """
        docs = self.retrieve(question)
        context = self._format_context(docs)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Replay recent conversation turns for context (multi-turn support).
        if history:
            for role, text in history[-6:]:  # last few turns is plenty
                if role == "user":
                    messages.append(HumanMessage(content=text))
                else:
                    messages.append(SystemMessage(content=f"(Earlier answer) {text}"))

        # The actual question, with the retrieved context attached.
        messages.append(HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {question}"
        ))

        response = self.llm.invoke(messages)
        answer_text = response.content.strip()

        # If the model said it couldn't find the answer, don't show sources
        # (there's nothing meaningful to cite).
        not_found = "could not find the answer" in answer_text.lower()
        sources = [] if not_found else self._format_sources(docs)

        return {"answer": answer_text, "sources": sources}
