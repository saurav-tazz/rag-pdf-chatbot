"""Probe FAISS retrieval scores for multiple questions.

Runs each query with several (k, score_threshold) configurations and prints
the scores + short content previews so we can tune retrieval relevance.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "data/rag-test-doc.pdf"

# -----------------------------#
# 1. Load + split
# -----------------------------#

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Loaded {len(documents)} pages -> {len(chunks)} chunks")

# -----------------------------#
# 2. Embeddings + index
# -----------------------------#

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

# -----------------------------#
# 3. Questions to probe
# -----------------------------#

QUESTIONS = [
    "Who invented the World Wide Web?",
    "Who wrote the first web browser?",
    "When was the first website created?",
    "What does HTML stand for?",
    "Where did the WWW invented?",  # intentionally rough phrasing
    "What is the capital of France?",  # out-of-domain check
    "Who was the first President of the United States?",  # out-of-domain check
]


# -----------------------------#
# 4. Probe with raw similarity_search_with_score
# -----------------------------#


def preview(text, n=140):
    t = "".join(ch for ch in text.replace("\n", " ").strip() if ord(ch) < 0x110000 and ord(ch) < 0x10000)
    return t[:n] + ("..." if len(t) > n else "")


def query_with_configs(query, configs):
    vec_query = vector_store.similarity_search_with_score(query, k=max(c["k"] for c in configs))

    print(f"\n========== QUERY: {query} ==========")
    rows = []
    for doc, score in vec_query:
        page_label = str(doc.metadata.get("page_label", doc.metadata.get("page", "?")))
        rows.append((score, doc, page_label))

    for cfg in configs:
        k = cfg["k"]
        thr = cfg["score_threshold"]
        kept = [(s, d, p) for s, d, p in rows[:k] if s <= thr]
        print(f"\n--- config k={k}, threshold={thr} -> {len(kept)} kept ---")
        for s, d, p in rows[:k]:
            mark = "KEEP" if s <= thr else "drop"
            print(f"  [{mark}] score={s:.4f} page={p}")
            print(f"      {ascii(preview(d.page_content))}")

    return rows


configs_default = [{"k": 3, "score_threshold": 1.30}]
configs_grid = [
    {"k": 3, "score_threshold": 1.10},
    {"k": 3, "score_threshold": 1.20},
    {"k": 3, "score_threshold": 1.30},
    {"k": 3, "score_threshold": 1.40},
    {"k": 4, "score_threshold": 1.30},
    {"k": 5, "score_threshold": 1.30},
    {"k": 5, "score_threshold": 1.20},
]

all_scores = {}
for q in QUESTIONS:
    rows = query_with_configs(q, configs_grid)
    top = rows[0][0] if rows else None
    best = min((s for s, _, _ in rows), default=None)
    all_scores[q] = (top, best, [s for s, _, _ in rows])

print("\n\n================ SCORE SUMMARY ================")
print(f"{'Query':<55} {'top1':>8} {'best':>8} n")
for q, (top, best, scores) in all_scores.items():
    print(f"{q:<55} {top:>8.4f} {best:>8.4f} {len(scores)}")
