from src.loader import load_and_split_pdf
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, retrieve_documents
from src.rag import get_llm, generate_answer


# -----------------------------
# Configuration
# -----------------------------

PDF_PATH = "data/rag-test-doc.pdf"


# -----------------------------
# Load and process PDF
# -----------------------------

chunks = load_and_split_pdf(PDF_PATH)

print(f"Created {len(chunks)} chunks.")


# -----------------------------
# Create embeddings
# -----------------------------

embedding_model = get_embedding_model()


# -----------------------------
# Create vector store
# -----------------------------

vector_store = create_vector_store(
    chunks,
    embedding_model
)

print("FAISS vector store created.")


# -----------------------------
# Create LLM
# -----------------------------

llm = get_llm()


# -----------------------------
# Ask question
# -----------------------------

question = "Who invented the World Wide Web?"


# -----------------------------
# Retrieve relevant documents
# -----------------------------

documents = retrieve_documents(
    vector_store,
    question,
    k=3
)

print(f"Retrieved {len(documents)} documents.")


# -----------------------------
# Generate answer
# -----------------------------

answer = generate_answer(
    llm,
    question,
    documents
)


# -----------------------------
# Display answer
# -----------------------------

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)
