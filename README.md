# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **strictly
from a provided PDF**. If the answer is not in the document, it says so. Every
answer shows the **page number(s)** it came from, so any claim can be traced back
to the source.

The ingested document is the **iPhone User Guide (iOS 7.1)**, a 162-page manual.
Page numbers shown in citations are **PDF page numbers** (1-based), which is what
you land on when you open the PDF to that page.

- **Orchestration:** LangChain 1.x
- **UI:** Streamlit
- **Vector DB:** Qdrant Cloud (free tier)
- **Embedding model:** FastEmbed `BAAI/bge-small-en-v1.5` (384 dims, runs locally, no API key)
- **Chat model:** Groq `llama-3.3-70b-versatile` (free API)

---

## Running the app

The vector index is **already populated in Qdrant Cloud**, so there is no
ingestion step to run. The app answers questions as soon as the container starts.

```bash
# 1. Clone
git clone <your-repo>
cd <repo>

# 2. Create your .env from the template and fill in the keys
cp .env.example .env
#    -> edit .env: GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 3. Build
docker build -t chatbot:1.0 .

# 4. Run  (the app listens on port 8501)
docker run -p 8501:8501 --env-file .env chatbot:1.0

# 5. Open the app
#    http://localhost:8501
```

**Port:** the application runs on **8501**. Map it with `-p 8501:8501` as shown.
(To change it, set `APP_PORT` in `.env` and adjust the `-p` mapping and the
`EXPOSE`/`--server.port` in the Dockerfile to match.)

No steps are required beyond the five above.

---

## Configuration & secrets

All configuration lives in `src/config.py` and is read from environment
variables. Secrets are **never** hardcoded or committed:

- `.env` is listed in `.gitignore` and is never in the repo or its history.
- `.env.example` documents every variable with a placeholder and description.
- The app fails fast with a clear message if a required key is missing.

---

## How it works

```
PDF ──ingest.py──>  [load pages] ──> [chunk] ──> [embed] ──> Qdrant Cloud
                                                                  │
User question ──app.py──> rag.py ──> [embed question] ──> [search] ──> top-K chunks
                                                                  │
                          [grounded prompt + chunks] ──> chat model ──> answer + page citations
```

### Ingestion (`src/ingest.py`)
Run once before submission: `python src/ingest.py path/to/document.pdf`

1. **Load** — `PyPDFLoader` reads the PDF into one document per page and
   captures the **page number** in metadata. We need that page number for the
   mandatory citations, so capturing it at load time is essential.
2. **Chunk** — see "Chunking strategy" below.
3. **Enrich metadata** — each chunk stores `page` (1-based), `source`
   (file name), and `chunk_index`.
4. **Embed & upload** — chunks are embedded with `text-embedding-3-small` and
   stored in a Qdrant collection (cosine distance, 1536-dim vectors).

### Retrieval & answering (`src/rag.py`)
- The question is embedded with the **same** model used at ingestion (so query
  and stored vectors share one space) and the top `TOP_K` (default 4) chunks
  are retrieved.
- A strict system prompt instructs the model to answer **only** from the
  retrieved context and to reply "I could not find the answer to that in the
  document." when the context doesn't contain it.
- `temperature=0` makes answers deterministic and discourages the model from
  filling gaps with outside knowledge.
- The page/source of each retrieved chunk is returned and shown in the UI.

### UI (`src/app.py`)
- Streamlit chat interface with **multi-turn** memory held in session state.
- Recent turns are passed to the model so follow-up questions resolve
  references like "it"/"that".
- Citations are displayed beneath every answer.

---

## Design decisions (for the technical interview)

**Chunking strategy.** `RecursiveCharacterTextSplitter`, `chunk_size=1000`
characters, `chunk_overlap=150`. Recursive splitting prefers natural
boundaries (paragraph → line → sentence → word), so chunks rarely break
mid-thought. 1000 characters is large enough to hold a coherent idea yet small
enough to keep retrieval precise and prompts focused. ~15% overlap preserves
context across boundaries so a sentence spanning two chunks isn't lost. These
are exposed as `CHUNK_SIZE`/`CHUNK_OVERLAP` so they can be tuned per document.

**Vector store metadata.** Each chunk stores `page`, `source`, and
`chunk_index`. `page` powers the required citations; `source` keeps answers
traceable if more than one document is ever ingested; `chunk_index` helps with
debugging and ordering. Note: `page` is the 1-based **PDF** page index from the
loader. The iPhone guide also prints its own page numbers in the page footer,
which are offset by the cover/contents pages — citations use the PDF index
because that is what a reader opening the PDF will navigate to.

**Embedding model.** FastEmbed `bge-small-en-v1.5` — a strong general-purpose
English retrieval model that runs **locally** via ONNX, with no API key, no GPU,
and no heavy PyTorch dependency, which keeps the Docker image small. It outputs
384-dimensional vectors. Because it's from Qdrant, it pairs naturally with the
vector store. The embedding model defines what "similar" means, so retrieval
quality is bounded by it; the same model is used at ingestion and at query time
so the question and stored vectors share one space. The choice is isolated in
`config.py`, so swapping to a larger model (or a hosted one) is a one-line change.

**Chat model.** Groq `llama-3.3-70b-versatile` — a high-quality open model served
on Groq's very fast, free API. `temperature=0` keeps answers grounded. Swappable
via `CHAT_MODEL`.

**Why LangChain (not full LangGraph).** This is a linear retrieve-then-generate
pipeline with simple session memory, which LangChain 1.x handles directly and
readably. LangGraph's stateful graph orchestration would add complexity without
benefit here; it would pay off for branching, tool-using, or long-running
agents.
