from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage

load_dotenv()

# --- Build the vector store (same as before) ---
loader = PyPDFLoader("data/Module_2_Full.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- Build or load the vector store, avoiding duplicates ---
if os.path.exists("chroma_db"):
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
else:
    print("Building new vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# --- Define our RAG search as a TOOL the agent can call ---
@tool
def search_dbms_notes(query: str) -> str:
    """Search the DBMS lecture notes (Relational Model, Keys, Constraints,
    Relational Algebra) to answer questions about database theory.
    Use this whenever the user asks a database/DBMS concept question."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the document."

    result = ""
    for doc in docs:
        page = doc.metadata.get("page_label", "unknown")
        result += f"[Page {page}]: {doc.page_content}\n\n"
    return result


# --- A second tool, just to prove the agent can choose between tools ---
@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression, e.g. '5 + 5' or '12 * 3'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# --- Build the agent with BOTH tools ---
llm = ChatGroq(model="openai/gpt-oss-120b")
tools = [search_dbms_notes, calculator]

system_prompt = """You are a DBMS teaching assistant. You have access to a tool called
search_dbms_notes that contains the user's actual lecture notes.

IMPORTANT RULE: For ANY question related to databases, DBMS, relational model,
keys, constraints, or relational algebra — you MUST call search_dbms_notes first,
even if you think you already know the answer. Always ground your answer in
the retrieved notes and mention it comes from the document.

Only skip search_dbms_notes for clearly unrelated requests (like calculations)."""

agent = create_react_agent(llm, tools, prompt=system_prompt)


# --- Test it ---
if __name__ == "__main__":
    user_input = "What is the difference between entity integrity and referential integrity? Also, what is 12 times 4?"

    for step in agent.stream({"messages": user_input}, stream_mode="values"):
        step["messages"][-1].pretty_print()