from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "data/rag-test-doc.pdf"

# Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Number of pages: {len(documents)}")


# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")


# Display the first 5 chunks
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)
    print(f"Metadata: {chunk.metadata}")