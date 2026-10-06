import os

import streamlit as st

from src.loader import load_and_split_pdf
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store, retrieve_documents
from src.rag import get_llm, generate_answer


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Cached resources
# -----------------------------

@st.cache_resource
def load_embedding_model():
    """Load the embedding model once."""

    return get_embedding_model()


@st.cache_resource
def load_llm():
    """Load the Llama model once."""

    return get_llm()


# -----------------------------
# Application title
# -----------------------------

st.title("📄 RAG PDF Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# -----------------------------
# Initialize session state
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# -----------------------------
# File uploader
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# -----------------------------
# Process uploaded PDF
# -----------------------------

if uploaded_file is not None:

    # Check whether this is a new PDF
    if (
        st.session_state.uploaded_filename
        != uploaded_file.name
    ):

        st.session_state.vector_store = None

        st.session_state.uploaded_filename = (
            uploaded_file.name
        )

        # Save uploaded PDF
        temp_pdf_path = os.path.join(
            "data",
            uploaded_file.name
        )

        with open(temp_pdf_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        # -----------------------------
        # Load and split PDF
        # -----------------------------

        with st.spinner("Processing PDF..."):

            chunks = load_and_split_pdf(
                temp_pdf_path
            )

        st.success(
            f"PDF processed! Created {len(chunks)} chunks."
        )

        # -----------------------------
        # Load embedding model
        # -----------------------------

        with st.spinner("Loading embedding model..."):

            embedding_model = load_embedding_model()

        # -----------------------------
        # Create FAISS vector store
        # -----------------------------

        with st.spinner("Building searchable index..."):

            st.session_state.vector_store = (
                create_vector_store(
                    chunks,
                    embedding_model
                )
            )

        st.success(
            "PDF is ready for questions!"
        )


# -----------------------------
# Ask questions
# -----------------------------

if st.session_state.vector_store is not None:

    question = st.text_input(
        "Ask a question about the PDF:"
    )

    if question:

        # -----------------------------
        # Retrieve relevant documents
        # -----------------------------

        with st.spinner("Searching the document..."):

            documents = retrieve_documents(
                st.session_state.vector_store,
                question,
                k=3
            )

        # -----------------------------
        # Load LLM
        # -----------------------------

        llm = load_llm()

        # -----------------------------
        # Generate answer
        # -----------------------------

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                llm,
                question,
                documents
            )

        # -----------------------------
        # Display answer
        # -----------------------------

        st.subheader("Answer")

        st.write(answer)
        