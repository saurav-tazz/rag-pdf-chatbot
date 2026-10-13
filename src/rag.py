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
You are a helpful assistant that explains information
from uploaded documents to students.

RULES:
1. Use ONLY the provided context to answer.
2. Explain the relevant information in clear, simple language.
3. When the context provides only a brief statement, explain
   its meaning without adding unsupported facts.
4. Do not merely copy a short phrase from the context.
5. If the context does not contain enough information to
   answer the question, say:
   "I could not find enough information in the document."
6. Keep the answer focused on the question.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Write a clear, useful answer:
"""

    response = llm.invoke(prompt)

    return response.content.strip()
