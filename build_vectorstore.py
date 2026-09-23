from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Load PDF
loader = PyPDFLoader("data/Module_2_Full.pdf")
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: Set up the embedding model (same one as before)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create the vector store — this embeds ALL chunks and stores them
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"   # saves to disk so we don't redo this every time
)

print(f"Stored {len(chunks)} chunks in the vector store.")

# Step 5: Test a similarity search directly
query = "What is a primary key?"
results = vectorstore.similarity_search(query, k=3)  # top 3 most relevant chunks

print(f"\n--- Top 3 chunks for query: '{query}' ---\n")
for i, doc in enumerate(results):
    print(f"Result {i+1} (page {doc.metadata['page_label']}):")
    print(doc.page_content)
    print("---")