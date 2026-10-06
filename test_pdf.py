from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/rag-test-doc.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Number of pages: {len(documents)}")

for i, document in enumerate(documents[:5]):
    print(f"\n--- Page {i + 1} ---")
    print(document.page_content[:500])