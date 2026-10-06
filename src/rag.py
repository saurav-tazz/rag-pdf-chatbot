from langchain_ollama import ChatOllama


def get_llm():
    """Create and return the Llama model."""

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


def generate_answer(llm, question, documents):
    """Generate an answer using retrieved document context."""

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

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

    response = llm.invoke(prompt)

    return response.content
