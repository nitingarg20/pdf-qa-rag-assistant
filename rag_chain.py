from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()  # loads GROQ_API_KEY from .env

# --- Steps 1-4: same as before (load, split, embed, store) ---
loader = PyPDFLoader("data/Module_2_Full.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Step 5: Turn the vectorstore into a "retriever" object
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 6: Set up the LLM
llm = ChatGroq(model="openai/gpt-oss-120b")

# Step 7: Build the prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the following context from the document.
If the answer isn't in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:
""")

# Step 8: Helper to format retrieved chunks into plain text for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 9: Build the full RAG chain using LangChain's pipe syntax
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 10: Ask a question!
question = "What is the capital of France?"
answer = rag_chain.invoke(question)

print(f"Question: {question}\n")
print(f"Answer: {answer}")