# Specification

## Purpose

Let a user ask natural-language questions about a PDF they supply and receive
short, grounded answers with visible evidence, entirely on the local machine.

## Scope

In scope: one PDF per session, question answering over that PDF, citation of the
chunks used, refusal when the document does not cover the question.

Out of scope: multiple documents at once, persistent chat history across restarts,
user accounts, hosted/cloud inference, OCR of scanned PDFs, editing PDFs, and
serving more than one user concurrently.

## Users

- **Student / reader** — wants explained answers from a document (course notes,
  papers, articles) without reading it end to end.
- **Developer** — runs the CLI demo or the probe scripts to check pipeline
  behavior and retrieval quality.

## Functional requirements

### FR-1 Upload a PDF

The UI accepts exactly one file with the `.pdf` extension through a Streamlit
file uploader. Non-PDF files are rejected by the uploader's type filter.

### FR-2 Process the document

On upload, the app writes the file to `data/<original filename>`, extracts text
per page, applies the fixed normalization table for known extraction artifacts,
and splits the text into ~500-character chunks with 50-character overlap. It
reports the number of pages processed implicitly through chunk count and shows a
success message with the chunk count.

### FR-3 Re-upload resets the session

Uploading a file with a different name discards the previous index and chat
history before processing. Uploading the same file again does not reprocess it.

### FR-4 Ask a question

Once processing succeeds, a chat input is available. Submitting a question adds
it to the transcript, retrieves candidates, and produces an answer in the same
turn.

### FR-5 Retrieval

Retrieval returns up to 3 chunks (`k=3`) whose L2 distance to the question
embedding is `<= 1.20`. Chunks outside the threshold are excluded. Zero returned
chunks is a valid, expected outcome.

### FR-6 Grounded answering

The answer must be derived only from the retrieved chunks. When retrieval returns
nothing, the answer is exactly:

> I could not find the answer in the document.

When chunks exist but do not contain enough information, the answer is:

> I could not find enough information in the document.

### FR-7 Evidence display

For every answered turn with at least one retained chunk, the UI shows a
`Retrieved Evidence` section with one expander per chunk, labelled with the file
basename and 1-based page number, containing the exact text supplied to the
model.

### FR-8 Transcript

The UI renders the full current-session transcript in order. Malformed history
entries are ignored rather than raised.

### FR-9 CLI equivalent

`app.py` runs the same pipeline non-interactively against
`data/rag-test-doc.pdf` with a fixed question and prints the question and answer,
for smoke testing without a browser.

## Non-functional requirements

- **NFR-1 Local-only.** No cloud AI service, no API keys, no telemetry. The only
  network activity is the one-time Hugging Face model download and the loopback
  Ollama call.
- **NFR-2 Reproducible.** `temperature=0`; the same question over the same PDF
  and models should produce the same or near-identical answer.
- **NFR-3 CPU-capable.** Embedding and indexing run on CPU.
- **NFR-4 Responsive feedback.** Long steps (PDF processing, model loading, index
  building, retrieval, generation) show a spinner or status message.
- **NFR-5 Model reuse.** The embedding model and LLM are constructed once per
  process, not once per question.
- **NFR-6 Privacy of content.** No document text or question is sent anywhere
  except the local Ollama process.

## Interfaces

### Python API (`src/`)

```python
load_and_split_pdf(pdf_path: str) -> list[Document]      # src/loader.py
get_embedding_model() -> HuggingFaceEmbeddings            # src/embeddings.py
create_vector_store(chunks, embedding_model) -> FAISS      # src/vector_store.py
retrieve_documents(vector_store, query, k=3, score_threshold=1.20) -> list[Document]
get_llm() -> ChatOllama                                   # src/rag.py
generate_answer(llm, question, documents) -> str          # src/rag.py
```

### Entry points

| Entry point | Command |
| --- | --- |
| Chat UI | `python -m streamlit run streamlit_app.py` |
| CLI demo | `python app.py` |
| Test suite | `python -m pytest -q` |

## Acceptance criteria

1. With `data/rag-test-doc.pdf` loaded, "Who invented the World Wide Web?"
   returns an answer naming Tim Berners-Lee and shows three evidence chunks.
2. The same question, asked after deleting the retained chunks (`k=0`), returns
   the "could not find the answer" string without calling the LLM.
3. "What is the capital of France?" returns no retained chunks and therefore the
   refusal string.
4. Uploading a second, differently named PDF clears the transcript and answers
   from the new document only.
5. `python -m pytest -q` passes on a machine with the sample PDF present.

## Assumptions and dependencies

- Ollama is installed and running, with `llama3.2:3b` pulled.
- The PDF has an extractable text layer (no OCR is performed).
- Dependencies in `requirements.txt` are installed in `.venv/`.
- The sample document `data/rag-test-doc.pdf` is present locally; it is
  git-ignored and must be supplied by the user.
