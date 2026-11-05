# RAG PDF Chatbot

Ask questions about a PDF and get answers grounded in its contents, running
entirely on your machine. Upload a document, the app embeds and indexes it, then
retrieves the most relevant passages for each question and passes them to a
local LLM. Every answer shows the exact passages it was based on.

## Features

- Local-only inference: no API keys, no cloud services, works offline after the
  one-time embedding model download.
- PDF text extraction, light artifact cleanup, and 500-character chunking with
  50-character overlap.
- FAISS similarity search with a distance threshold, so unrelated questions
  produce a refusal instead of a hallucinated answer.
- Chat UI with the retrieved evidence shown for every answer.
- CLI demo for quick pipeline checks.

## Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.12 |
| Orchestration | LangChain 1.x (`langchain-community`, `langchain-huggingface`, `langchain-ollama`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU) |
| Vector store | FAISS (`faiss-cpu`) |
| LLM | `llama3.2:3b` via Ollama, `temperature=0` |
| UI | Streamlit |

## Prerequisites

- Python 3.12
- [Ollama](https://ollama.com) installed and running, with the model pulled:
  `ollama pull llama3.2:3b`
- A PDF with an extractable text layer (no OCR is performed)

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

Put your PDF in `data/` if you want to use the CLI demo (`data/*.pdf` is
git-ignored). The sample document used by the tests and `app.py` is
`data/rag-test-doc.pdf`.

## Usage

```bash
# Interactive chat UI at http://localhost:8501
.venv/Scripts/python.exe -m streamlit run streamlit_app.py

# CLI demo: one hard-coded question over data/rag-test-doc.pdf
.venv/Scripts/python.exe app.py
```

In the UI, upload a PDF, wait for the "PDF is ready for questions!" message, then
ask questions in the chat box. Uploading a different file starts a fresh session.

## Project structure

```
app.py                  CLI demo of the full pipeline
streamlit_app.py        Streamlit chat UI
src/loader.py           PDF loading, text normalization, chunking
src/embeddings.py       Hugging Face embedding model factory
src/vector_store.py     FAISS index creation + thresholded retrieval
src/rag.py              Ollama LLM factory, prompt, answer generation
tests/                  pytest retrieval regression suite
test_*.py               Manual probe/debug scripts (not collected by pytest)
docs/                   Specification, architecture, testing notes
```

## Testing

```bash
.venv/Scripts/python.exe -m pytest -q
```

The pytest suite needs `data/rag-test-doc.pdf` and downloads the embedding model
on first run; it does not require Ollama. See [docs/testing.md](docs/testing.md)
for the manual probe scripts and what each one covers.

## Documentation

- [docs/spec.md](docs/spec.md) — functional and non-functional requirements,
  interfaces, acceptance criteria
- [docs/architecture.md](docs/architecture.md) — components, request flow,
  retrieval threshold calibration, design decisions, limitations
- [docs/testing.md](docs/testing.md) — test layout, prerequisites, coverage
- [AGENTS.md](AGENTS.md) — commands and conventions for AI coding agents

## How retrieval works

Chunks are embedded with MiniLM and searched with FAISS, which returns L2
distances (lower is better). Only chunks with a distance of `1.20` or less are
kept, up to 3. That threshold was tuned on the bundled article: on-topic
questions land around 0.36–1.06, while off-topic questions stay at 1.63 or
above. When nothing passes, the assistant replies
"I could not find the answer in the document."

## Limitations

- One PDF per session; no multi-document search.
- No conversation memory — follow-up questions do not see earlier turns.
- The index is rebuilt on every upload and is not persisted.
- Uploaded files are written to `data/` and are not cleaned up.
- Tuned for a single local user; there is no authentication or multi-user
  isolation.
- Retrieval quality is bounded by MiniLM; heavy paraphrasing may trigger a
  refusal.

## Status

Under active development.
