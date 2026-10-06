from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# 1. Load PDF
# -----------------------------

pdf_path = "data/rag-test-doc.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Number of pages: {len(documents)}")


# -----------------------------
# 2. Split into chunks
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# -----------------------------
# 3. Create embedding model
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Create FAISS vector store
# -----------------------------

vector_store = FAISS.from_documents(
    chunks,
    embedding_model
)

print("FAISS vector store created successfully.")


# -----------------------------
# 5. Search the vector store
# -----------------------------

query = "Who invented the World Wide Web?"

results = vector_store.similarity_search(
    query,
    k=3
)


# -----------------------------
# 6. Display results
# -----------------------------

print(f"\nQuery: {query}")

for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print(f"Page: {result.metadata.get('page_label')}")
    