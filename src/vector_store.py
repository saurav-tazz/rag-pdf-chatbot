from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):
    """Create a FAISS vector store from document chunks."""

    return FAISS.from_documents(
        chunks,
        embedding_model
    )


def retrieve_documents(
    vector_store,
    query,
    k=3,
    score_threshold=1.30
):
    """Retrieve relevant documents using a distance threshold."""

    documents_with_scores = (
        vector_store.similarity_search_with_score(
            query,
            k=k
        )
    )

    relevant_documents = []

    for document, score in documents_with_scores:

        if score <= score_threshold:
            relevant_documents.append(document)

    return relevant_documents
