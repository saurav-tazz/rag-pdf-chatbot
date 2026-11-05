# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this project is

A local, offline Retrieval-Augmented Generation (RAG) chatbot over PDFs. A user
uploads a PDF in a Streamlit UI, the document is chunked and embedded with
`sentence-transformers/all-MiniLM-L6-v2`, indexed in FAISS, and questions are
answered by `llama3.2:3b` running in Ollama. Answers are constrained to the
retrieved context and the retrieved chunks are shown to the user as evidence.

Start here: [`docs/spec.md`](docs/spec.md) (behavior), then
[`docs/architecture.md`](docs/architecture.md) (how it is built).

## Environment

- Windows shell is **bash (Git Bash)**. Use POSIX commands (`ls`, `rm`, `cp`),
  forward-slash paths, and never `cmd.exe`/PowerShell syntax.
- Python **3.12**, virtualenv already present at `.venv/`.
- Interpreter: `.venv/Scripts/python.exe` (posix: `.venv/bin/python`).
- Ollama must be running locally with the `llama3.2:3b` model pulled
  (`ollama pull llama3.2:3b`). Everything else runs on CPU.

## Common commands

```bash
# Install dependencies (once)
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Run the Streamlit chat UI (http://localhost:8501)
.venv/Scripts/python.exe -m streamlit run streamlit_app.py

# Run the CLI smoke demo (uses data/rag-test-doc.pdf)
.venv/Scripts/python.exe app.py

# Retrieval regression tests (needs data/rag-test-doc.pdf)
.venv/Scripts/python.exe -m pytest -q
```

## Layout and module boundaries

| Path | Role |
| --- | --- |
| `src/loader.py` | PDF loading, text normalization, chunking (`load_and_split_pdf`) |
| `src/embeddings.py` | Embedding model factory (`get_embedding_model`) |
| `src/vector_store.py` | FAISS index creation + thresholded retrieval |
| `src/rag.py` | LLM factory (`get_llm`) + prompt + `generate_answer` |
| `streamlit_app.py` | Chat UI, session state, evidence display |
| `app.py` | Hard-coded CLI demo of the same pipeline |
| `tests/` | pytest suite (see `docs/testing.md`) |
| `test_*.py` (repo root) | Manual probe/debug scripts, **not** pytest tests |

Rules:

- Keep pipeline stages in their own `src/` module. UI files orchestrate; they do
  not contain retrieval or prompting logic.
- `streamlit_app.py` must keep loading the embedding model and LLM through the
  `@st.cache_resource` wrappers. Never construct those models per interaction.
- New runtime dependencies go into `requirements.txt` (UTF-8, one pinned
  `package==version` per line).
- Source files use CRLF line endings. Keep them consistent (do not normalize the
  whole repo to LF in an unrelated change).

## Retrieval rules (read before touching scores)

- FAISS returns **L2 distance**: lower is better. Default retrieval is `k=3` with
  `score_threshold=1.20` in `src/vector_store.py`.
- That threshold is calibrated to the bundled `data/rag-test-doc.pdf` and the
  MiniLM embedding model. Any change to chunking, the embedding model, the
  prompt, or the threshold must be validated with
  `tests/test_retrieval_relevance.py` and recorded in
  `docs/architecture.md`.
- `retrieve_documents` prints scores and chunk previews to stdout. This is
  intentional debugging output; `tests/test_retrieval_relevance.py` depends on
  the returned list, not the printing, but keep the diagnostics useful.
- `src/loader.py` contains a small table of PDF-extraction fixes (e.g.
  `Berners -Lee` → `Berners-Lee`). Add entries only for artifacts actually
  observed in a source PDF, with the observed text included, and never as a
  general-purpose spell checker.

## Answer behavior

`generate_answer` must: use only retrieved context, explain rather than echo a
short phrase, and return the refusal string
`"I could not find enough information in the document."` when context is
insufficient. It already returns
`"I could not find the answer in the document."` when nothing was retrieved.
Keeping these two messages distinct is expected behavior — do not collapse them.

## Data and secrets

- `data/*.pdf` is git-ignored, as are `faiss_index/`, `.venv/`, and
  `.streamlit/`. Never commit PDFs, uploads, or local indexes.
- Uploaded files are written to `data/<original name>` and **overwrite** a file
  of the same name. Be careful when testing with a file that shares the name of
  `data/rag-test-doc.pdf`.
- There are no API keys or cloud services in the pipeline. Do not introduce a
  hosted LLM or embedding provider without an explicit request.

## Verification expectations

- Change in `loader.py` / `vector_store.py` / `rag.py` → run
  `.venv/Scripts/python.exe -m pytest -q` (requires the sample PDF and the
  Ollama model only for answer-generation paths; the pytest suite is
  embedding-only and does not need Ollama).
- UI-only change → launch Streamlit and confirm the app still starts.
- Do not "fix" failing retrieval tests by moving the threshold until counts
  match; either the retrieval change is wrong or the expectation in
  `tests/test_retrieval_relevance.py` plus the threshold table in the docs must
  be updated deliberately, with the measured scores quoted.
