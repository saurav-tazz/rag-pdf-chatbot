from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """Create a FAISS vector store from document chunks."""

    return FAISS.from_documents(
        chunks,
        embedding_model
    )


def retrieve_documents(vector_store, query, k=3):
    """Retrieve the most relevant documents for a query."""

    return vector_store.similarity_search(
        query,
        k=k
    )
