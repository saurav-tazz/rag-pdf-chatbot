# Testing

## Layout

The repository has two kinds of test artifacts. `pytest.ini` restricts collection
to `tests/` so the manual scripts at the repo root are not treated as tests.

| Path | Kind | Purpose |
| --- | --- | --- |
| `tests/test_retrieval_relevance.py` | pytest | Regression tests for the tuned retrieval threshold |
| `tests/__init__.py` | pytest | Makes `tests` a package |
| `test_retrieval_verify.py` | script | Prints kept/dropped scores for 10 queries |
| `test_retrieval_probe.py` | script | Grid-searches `(k, threshold)` configurations and prints a score summary |
| `test_pdf.py` | script | Prints page/chunk counts and the first 5 chunks |
| `test_embeddings.py` | script | Cosine similarity sanity check for MiniLM |
| `test_faiss.py` | script | Builds an index and prints top-3 chunks for one query |
| `test_llm.py` | script | Sends one prompt to Ollama and prints the reply |
| `test_rag.py` | script | End-to-end manual pipeline without the threshold |

## Prerequisites

- `.venv/` with `requirements.txt` installed.
- `data/rag-test-doc.pdf` present (git-ignored, not shipped).
- The pytest suite needs no Ollama — it exercises loading, chunking, embedding,
  FAISS, and the default `retrieve_documents` threshold. The root scripts
  `test_llm.py` and `test_rag.py` do need Ollama with `llama3.2:3b`.
- First embedding run downloads `sentence-transformers/all-MiniLM-L6-v2`
  (~90 MB) and needs network access for that download only.

## Running

```bash
# Regression tests
.venv/Scripts/python.exe -m pytest -q

# One file
.venv/Scripts/python.exe -m pytest tests/test_retrieval_relevance.py -q

# Manual probes (each rebuilds the index and prints diagnostics)
.venv/Scripts/python.exe test_retrieval_verify.py
.venv/Scripts/python.exe test_retrieval_probe.py
```

## What the pytest suite asserts

Each test rebuilds the index (PDF → chunks → MiniLM → FAISS) and applies
`score_threshold=1.20` to the top-3 results.

| Test | Expectation |
| --- | --- |
| `test_www_inventor_keeps_top3` | 3 chunks kept, ascending scores, contains "world wide web" and "berners"; index has >10 chunks |
| `test_first_browser_keeps_top3` | 3 kept |
| `test_first_website_keeps_top3` | 3 kept |
| `test_html_definition_keeps_top3` | 3 kept |
| `test_borderline_query_keeps_two_relevant_chunks` | 2 kept for rough phrasing; "world wide web" present |
| `test_out_of_domain_queries_return_nothing` | 0 kept for Paris / first US President |
| `test_default_retrieve_documents_uses_new_threshold` | public helper returns 3 for the WWW question |

These are deliberately data-coupled assertions: chunk counts depend on the
specific PDF, the splitter settings, and the embedding model. If any of those
change, expect to update both the tests and the calibration table in
`docs/architecture.md` with freshly measured scores.

Runtime is dominated by repeated index construction (each test builds its own)
and embedding-model loading; expect tens of seconds on CPU.

## Manual checks not covered by automation

- Streamlit UI: upload → chunk count message → ask → answer plus evidence
  expanders → upload a different PDF resets the transcript.
- Ollama availability and answer quality (`test_llm.py`, `app.py`).
- Behavior with scanned/image-only PDFs (expected: few or no chunks).
