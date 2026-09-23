from langchain_huggingface import HuggingFaceEmbeddings

# Load a small, fast, well-regarded open-source embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Test it on two related sentences and one unrelated sentence
text1 = "A primary key uniquely identifies each row in a table."
text2 = "Every tuple in a relation must have a unique identifier."
text3 = "The weather in New York is sunny today."

vec1 = embedding_model.embed_query(text1)
vec2 = embedding_model.embed_query(text2)
vec3 = embedding_model.embed_query(text3)

print(f"Vector length: {len(vec1)}")  # how many numbers represent each chunk

# Manually compute cosine similarity to prove the concept
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\nSimilarity (related sentences): {cosine_similarity(vec1, vec2):.4f}")
print(f"Similarity (unrelated sentences): {cosine_similarity(vec1, vec3):.4f}")