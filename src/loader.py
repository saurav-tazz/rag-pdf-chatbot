import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path):
    """Load a PDF and split it into smaller chunks."""

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Light normalization of well-known PDF text-extraction artifacts.
    # PyPDFLoader (pypdf) sometimes inserts a space where a word was split
    # across a column break or hyphenation, for example "Berners -Lee",
    # "sci entist", and "star ts". Fix only the tokens we have confirmed
    # are broken in this PDF so we don't silently introduce misspellings.
    _NORMALIZATIONS = [
        (r"\bBerners -Lee\b", "Berners-Lee"),
        (r"\buser -friendly\b", "user-friendly"),
        (r"\bdot -com\b", "dot-com"),
        (r"\bweb -based\b", "web-based"),
        (r"\bsci entist\b", "scientist"),
        (r"\bstar ts\b", "starts"),
    ]

    for document in documents:
        page = document.page_content
        for pattern, replacement in _NORMALIZATIONS:
            new_page, n = re.subn(pattern, replacement, page)
            if n:
                page = new_page
        if page != document.page_content:
            document.page_content = page

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    return chunks
