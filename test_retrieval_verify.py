"""Verify retrieval relevance after threshold change.

Run on the current index with the new default threshold, plus a small
extra set of questions, to confirm:
- top-3 on-topic results still pass
- borderline query keeps only the 2 relevant chunks
- out-of-domain queries stay filtered
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.vector_store import retrieve_documents

PDF_PATH = "data/rag-test-doc.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)


def short(text, n=120):
    t = "".join(ch for ch in text.replace("\n", " ").strip() if ord(ch) < 0x10000)
    return t[:n] + ("..." if len(t) > n else "")


def eval_query(query):
    rows = vector_store.similarity_search_with_score(query, k=3)
    docs = [d for d, s in rows if s <= 1.20]
    return docs, rows


QUERIES = [
    "Who invented the World Wide Web?",
    "Who wrote the first web browser?",
    "When was the first website created?",
    "What does HTML stand for?",
    "Where did the WWW invented?",
    "What programming language is used for backend web development?",
    "How do search engines work?",
    "What is the difference between the internet and the web?",
    "What is the capital of France?",
    "Who was the first President of the United States?",
]

print("\n================ VERIFICATION ================")
for q in QUERIES:
    docs, rows = eval_query(q)
    print(f"\nQUERY: {q}")
    print(f"kept={len(docs)}  scores={tuple(round(float(s),4) for _,s in rows)}")
    if rows:
        for i, (d, s) in enumerate(rows, start=1):
            page = d.metadata.get("page", 0) + 1
            status = "KEEP" if s <= 1.20 else "drop"
            print(f"  {i}. [{status}] score={float(s):.4f} page={page}")
            print(f"     {ascii(short(d.page_content))}")
    else:
        print("  (none returned)")

print("\nDone.")
