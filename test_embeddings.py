from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts = [
    "The Internet connects computers around the world.",
    "The Internet is a global network of interconnected computers.",
    "I like eating pizza."
]


vectors = embedding_model.embed_documents(texts)

similarity_1_2 = cosine_similarity(
    [vectors[0]],
    [vectors[1]]
)[0][0]

similarity_1_3 = cosine_similarity(
    [vectors[0]],
    [vectors[2]]
)[0][0]

print(f"\nSimilarity between sentence 1 and 2: {similarity_1_2:.4f}")
print(f"Similarity between sentence 1 and 3: {similarity_1_3:.4f}")

print(f"Number of vectors: {len(vectors)}")
print(f"Vector dimensions: {len(vectors[0])}")

for i, vector in enumerate(vectors):
    print(f"\nVector {i + 1}:")
    print(vector[:10])