from langchain_ollama import ChatOllama


# -----------------------------
# 1. Create Llama model
# -----------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# -----------------------------
# 2. Ask a question
# -----------------------------

question = "What is Retrieval-Augmented Generation? Explain it simply."

response = llm.invoke(question)


# -----------------------------
# 3. Display response
# -----------------------------

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(response.content)