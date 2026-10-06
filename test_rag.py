from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama


# -----------------------------
# 1. Load PDF
# -----------------------------

pdf_path = "data/rag-test-doc.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Number of pages: {len(documents)}")


# -----------------------------
# 2. Split PDF into chunks
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
# 5. Ask a question
# -----------------------------

question = "What is the capital of France?"


# -----------------------------
# 6. Retrieve relevant chunks
# -----------------------------

results = vector_store.similarity_search(
    question,
    k=3
)

print(f"\nQuestion: {question}")
print(f"Retrieved {len(results)} relevant chunks.")


# -----------------------------
# 7. Combine retrieved chunks
# -----------------------------

context = "\n\n".join(
    result.page_content
    for result in results
)


# -----------------------------
# 8. Create Llama model
# -----------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# -----------------------------
# 9. Create RAG prompt
# -----------------------------

prompt = f"""
You are a helpful assistant answering questions based on a provided document.

Use ONLY the information contained in the context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# 10. Ask Llama
# -----------------------------

response = llm.invoke(prompt)


# -----------------------------
# 11. Display answer
# -----------------------------

print("\nAnswer:")
print(response.content)
