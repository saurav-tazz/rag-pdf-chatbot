from langchain_ollama import ChatOllama


def get_llm():
    """Create and return the Llama model."""

    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


def generate_answer(llm, question, documents):
    """Generate an answer using retrieved document context."""

    if not documents:
        return "I could not find the answer in the document."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information contained
in the provided context.

Do NOT use your own general knowledge.

If the answer is not explicitly supported by the context,
respond exactly with:

"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content
