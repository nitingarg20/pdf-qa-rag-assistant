from langchain_community.document_loaders import PyPDFLoader

# Step 1: Point the loader at our PDF
loader = PyPDFLoader("data/Module_2_Full.pdf")

# Step 2: Load it — this reads the PDF and splits it into one "Document" per page
documents = loader.load()

# Step 3: Inspect what we got
print(f"Total pages loaded: {len(documents)}")
print("\n--- Content of Page 1 ---\n")
print(documents[0].page_content)
print("\n--- Metadata of Page 1 ---\n")
print(documents[0].metadata)