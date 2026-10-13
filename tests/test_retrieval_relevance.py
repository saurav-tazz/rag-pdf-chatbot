"""Retrieval relevance checks for the current PDF and embedding stack.

These assertions are parameterized on the current document and the
sentence-transformers/all-MiniLM-L6-v2 embedding model used by the
project. They verify that:

- On-topic questions keep exactly the top-3 chunks under the default
  threshold.
- A borderline query keeps only the two clearly relevant chunks.
- Out-of-domain questions return no chunks.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = Path("data/rag-test-doc.pdf")
BACKEND_THRESHOLD = 1.20


def _index__():
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, model), len(chunks)


def _top_k_scores(store, query, k=3):
    rows = store.similarity_search_with_score(query, k=k)
    return [(float(s), d) for d, s in rows]


def _below_threshold(rows, threshold):
    return [(s, d) for s, d in rows if s <= threshold]


def test_www_inventor_keeps_top3():
    store, n_chunks = _index__()
    assert n_chunks > 10, f"expected >10 chunks, got {n_chunks}"

    rows = _top_k_scores(store, "Who invented the World Wide Web?")
    kept = _below_threshold(rows, BACKEND_THRESHOLD)

    assert len(kept) == 3, f"expected 3 kept, got {len(kept)}"
    assert kept[0][0] < kept[1][0] < kept[2][0], "scores should be sorted ascending"

    texts = " ".join(d.page_content.lower() for _, d in kept)
    for needle in ("world wide web", "berners"):
        assert needle in texts, f"missing expected term '{needle}' in kept chunks"


def test_first_browser_keeps_top3():
    store, _ = _index__()
    rows = _top_k_scores(store, "Who wrote the first web browser?")
    kept = _below_threshold(rows, BACKEND_THRESHOLD)

    assert len(kept) == 3, f"expected 3 kept, got {len(kept)}"


def test_first_website_keeps_top3():
    store, _ = _index__()
    rows = _top_k_scores(store, "When was the first website created?")
    kept = _below_threshold(rows, BACKEND_THRESHOLD)

    assert len(kept) == 3, f"expected 3 kept, got {len(kept)}"


def test_html_definition_keeps_top3():
    store, _ = _index__()
    rows = _top_k_scores(store, "What does HTML stand for?")
    kept = _below_threshold(rows, BACKEND_THRESHOLD)

    assert len(kept) == 3, f"expected 3 kept, got {len(kept)}"


def test_borderline_query_keeps_two_relevant_chunks():
    """Rough phrasing should drop the weaker third chunk, not the two good ones.

    At the current threshold this query keeps 2 documents, not 3, because
    the third candidate is about the broader 'internet' rather than the WWW.
    """
    store, _ = _index__()
    rows = _top_k_scores(store, "Where did the WWW invented?")
    kept = _below_threshold(rows, BACKEND_THRESHOLD)

    assert len(kept) == 2, f"expected 2 kept for borderline query, got {len(kept)}"

    texts = " ".join(d.page_content.lower() for _, d in kept)
    assert "world wide web" in texts


def test_out_of_domain_queries_return_nothing():
    store, _ = _index__()
    for query in (
        "What is the capital of France?",
        "Who was the first President of the United States?",
    ):
        rows = _top_k_scores(store, query)
        kept = _below_threshold(rows, BACKEND_THRESHOLD)
        assert len(kept) == 0, f"expected 0 kept for OOD query, got {len(kept)}"


def test_default_retrieve_documents_uses_new_threshold():
    """The public retrieve helper should match the threshold we tuned."""
    from src.vector_store import retrieve_documents

    store, _ = _index__()
    docs = retrieve_documents(store, "Who invented the World Wide Web?", k=3)
    assert len(docs) == 3
