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
    score_threshold=1.20
):
    """Retrieve relevant documents using a distance threshold.

    Threshold chosen by probing multiple questions on the current PDF /
    embedding model (sentence-transformers/all-MiniLM-L6-v2). With this
    index the top-3 results for on-topic queries score between ~0.36 and
    ~1.06, while out-of-domain queries only reach ~1.63+. Threshold 1.20
    keeps the two clearly relevant chunks for borderline queries without
    pulling in the weaker third chunk that appears at 1.30.
    """

    documents_with_scores = (
        vector_store.similarity_search_with_score(
            query,
            k=k
        )
    )

    print(f"\n========== QUERY: {query} ==========")

    relevant_documents = []

    for document, score in documents_with_scores:

        page = document.metadata.get(
            "page",
            0
        ) + 1

        safe = "".join(ch for ch in document.page_content[:800] if ord(ch) < 128 or ch in "\n\r\t")
        print(f"\nScore: {score:.4f}")
        print(f"Page: {page}")
        print("Content:")
        print(safe if safe.strip() else "(non-ascii content omitted)")

        if score <= score_threshold:
            relevant_documents.append(document)

    print(
        f"\nRelevant documents: "
        f"{len(relevant_documents)}"
    )

    return relevant_documents
