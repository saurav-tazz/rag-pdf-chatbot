# Architecture

## Overview

The application is a single-process, fully local RAG pipeline. There is no
server component beyond the Streamlit process, no database, and no network call
except the loopback call to Ollama and the one-time Hugging Face model download.

```
                 ┌──────────────────────────┐
   user upload   │  streamlit_app.py (UI)   │  user question
   ─────────────▶│  session state, chat,    │◀──────────────
                 │  evidence expanders      │
                 └───────┬──────────┬───────┘
                         │          │
         ┌───────────────▼──┐   ┌───▼────────────────┐
         │ src/loader.py    │   │ src/vector_store.py │
         │ PDF → documents  │   │ FAISS search +      │
         │ → 500-char chunks│   │ distance threshold  │
         └───────────────┬──┘   └───▲────────┬────────┘
                         │              │    │
                 ┌───────▼──────────────┴──┐ │  top-k chunks
                 │ src/embeddings.py       │ │
                 │ all-MiniLM-L6-v2 (384d) │ │
                 └─────────────────────────┘ │
                                             ▼
                                   ┌────────────────────┐
                                   │ src/rag.py         │
                                   │ prompt + ChatOllama│
                                   │ llama3.2:3b, T=0   │
                                   └─────────┬──────────┘
                                             ▼
                                     grounded answer
```

## Components

### `src/loader.py` — ingestion

- `load_and_split_pdf(pdf_path) -> list[Document]`
- Uses `PyPDFLoader` (pypdf). Each `Document` carries `source` and `page`
  metadata. `page` is **0-indexed**; the UI displays `page + 1`.
- Applies a narrow, hard-coded normalization table for text-extraction artifacts
  seen in the bundled PDF (`Berners -Lee`, `user -friendly`, `dot -com`,
  `web -based`, `sci entist`, `star ts`). Only confirmed artifacts are listed.
- Splits with `RecursiveCharacterTextSplitter(chunk_size=500,
  chunk_overlap=50)` — roughly one to three paragraphs per chunk, enough context
  for a MiniLM embedding without drowning a single fact.

### `src/embeddings.py` — vectors

- `get_embedding_model()` returns `HuggingFaceEmbeddings` wrapping
  `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU, ~90 MB).
- The model is downloaded on first use and cached by the Hugging Face cache.
- Chosen for speed and small footprint; it is not a strong retriever for long
  documents, which is why retrieval is threshold-gated (below).

### `src/vector_store.py` — index and retrieval

- `create_vector_store(chunks, embedding_model)` → in-memory
  `FAISS.from_documents(...)`, rebuilt on every upload. Nothing is persisted to
  disk (`faiss_index/` is git-ignored but currently unused).
- `retrieve_documents(vector_store, query, k=3, score_threshold=1.20)`:
  - calls `similarity_search_with_score(query, k=k)`;
  - **FAISS scores here are L2 distances — lower is better;**
  - keeps only chunks with `score <= score_threshold`;
  - prints each candidate's score, 1-based page, and an ASCII-safe preview to
    stdout for debugging.

### `src/rag.py` — generation

- `get_llm()` returns `ChatOllama(model="llama3.2:3b", temperature=0)`.
- `generate_answer(llm, question, documents)`:
  - returns `"I could not find the answer in the document."` if `documents` is
    empty (no LLM call);
  - joins chunk contents with blank lines into a `DOCUMENT CONTEXT` block;
  - uses a rule-based prompt: answer only from context, explain rather than
    copy short phrases, do not add unsupported facts, and reply
    `"I could not find enough information in the document."` when the context is
    insufficient.

### `streamlit_app.py` — interactive surface

- `@st.cache_resource` wrappers keep one embedding model and one LLM per server
  process.
- Session state keys: `vector_store`, `uploaded_filename`, `messages`.
- Uploading a *different* filename clears the index and the chat history, writes
  the upload to `data/<filename>`, then chunks → embeds → indexes with progress
  spinners.
- Each turn retrieves `k=3`, generates an answer, then renders
  `### Retrieved Evidence` expanders labelled `Evidence N: <file> — Page <n>`
  with the exact text passed to the model.
- The chat transcript is stored as `{"role", "content"}` dicts and defensively
  skipped if malformed.

### `app.py` — CLI demo

Non-interactive smoke path: loads `data/rag-test-doc.pdf`, builds the index,
asks one hard-coded question, prints retrieved count and the answer.

## Request flow (chat turn)

1. User submits a question via `st.chat_input`.
2. `retrieve_documents` embeds the question and searches the FAISS index for the
   3 nearest chunks; candidates beyond distance `1.20` are discarded.
3. If nothing survives, `generate_answer` returns the "could not find the
   answer" message without invoking the LLM.
4. Otherwise the surviving chunks are concatenated and sent with the question to
   `llama3.2:3b`.
5. The answer is displayed and appended to history; evidence chunks render in
   expanders beneath it.

## Retrieval calibration

The threshold is empirical, measured against `data/rag-test-doc.pdf` (a
Wikipedia-style article on the World Wide Web) with MiniLM embeddings. Measured
top-3 behaviour at `score_threshold=1.20`:

| Query | Kept | Notes |
| --- | --- | --- |
| Who invented the World Wide Web? | 3 | best match ≈ 0.36 |
| Who wrote the first web browser? | 3 | |
| When was the first website created? | 3 | |
| What does HTML stand for? | 3 | |
| Where did the WWW invented? (rough phrasing) | 2 | third chunk is about the Internet broadly, ≈ 1.30 |
| What is the capital of France? | 0 | out-of-domain, ≈ 1.63+ |
| Who was the first President of the United States? | 0 | out-of-domain |

Consequences of the design:

- On-topic queries keep up to 3 chunks in the ~0.36–1.06 range.
- Out-of-domain queries keep nothing, so the assistant refuses instead of
  hallucinating.
- The numbers are corpus- and model-specific. Changing the embedding model,
  chunk size, or the source document invalidates the table; re-run
  `test_retrieval_probe.py` / `test_retrieval_verify.py` and update both the
  default in `src/vector_store.py` and this table.

## Design decisions

| Decision | Rationale |
| --- | --- |
| Local Ollama instead of a hosted API | No cost, no keys, works offline; acceptable latency for one user |
| In-memory FAISS per upload | Simplest correct model for a single-document chat session |
| Distance threshold instead of fixed top-k only | Prevents answering unrelated questions with weakly related text |
| Evidence shown to the user | Makes grounding auditable and exposes retrieval failures |
| `temperature=0` | Factual, reproducible answers over the retrieved context |
| Prompt-level grounding rules | The model is small; explicit "use only the context" rules reduce drift |
| Single shared prompt builder | UI and CLI produce identical answers for the same question |

## Known limitations

- One document per session; no multi-document corpora or per-file indexes.
- Uploads are written into `data/` and never cleaned up.
- No conversation memory is passed to the LLM — follow-up questions are
  answered from the current question's retrieval only.
- The index is rebuilt from scratch on each upload (no caching or persistence).
- `page` metadata is 0-indexed in the store and converted only for display.
- Retrieval quality depends on MiniLM; paraphrase-heavy or very long questions
  can fall outside the threshold and produce a refusal.
- No authentication, rate limiting, or streaming responses; the Streamlit app is
  a single-user local tool.
