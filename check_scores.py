import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use your existing persisted vector store — adjust the hash folder name if needed
# Find your actual folder name by running: dir chroma_db_*
persist_dir = [d for d in os.listdir(".") if d.startswith("chroma_db_")][0]
print(f"Using vector store: {persist_dir}")

vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

queries = [
    "What is candidate key",
    "What is capital of France",
    "How do I write a SQL query to create a table",
    "What is referential integrity",
    "Explain the JOIN operation",
]

for q in queries:
    print(f"\n--- Query: '{q}' ---")
    results = vectorstore.similarity_search_with_score(q, k=3)
    for doc, distance in results:
        page = doc.metadata.get("page_label", "?")
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"  distance={distance:.4f}  page={page}  preview='{preview}...'")