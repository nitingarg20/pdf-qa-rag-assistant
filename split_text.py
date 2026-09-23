import streamlit as st
import os
import tempfile
import hashlib
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📘")
st.title("📘 PDF Q&A Assistant")
st.caption("Upload any PDF and ask questions grounded in its content")


# --- Cache the embedding model separately — this is the SAME for every file ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- Build a vector store + agent for a SPECIFIC uploaded file ---
# Cached per file_hash — same file won't rebuild, different file will
@st.cache_resource
def build_agent_for_file(file_hash: str, file_path: str):
    embedding_model = get_embedding_model()

    persist_dir = f"chroma_db_{file_hash}"

    if os.path.exists(persist_dir):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embedding_model, persist_directory=persist_dir
        )

    # --- Relevance-scored retrieval: we decide relevance ourselves, not the LLM ---
    RELEVANCE_THRESHOLD = 0.5  # lower distance = more similar; tune based on testing

    def get_relevant_context(query: str):
        """Returns (context_string, is_relevant: bool) using actual similarity scores."""
        results = vectorstore.similarity_search_with_score(query, k=3)
        if not results:
            return "", False

        # Chroma returns (doc, distance) — LOWER distance means MORE similar
        best_distance = results[0][1]
        is_relevant = best_distance < RELEVANCE_THRESHOLD

        context = ""
        for doc, distance in results:
            page = doc.metadata.get("page_label", "unknown")
            context += f"[Page {page}]: {doc.page_content}\n\n"
        return context, is_relevant

    @tool
    def search_document(query: str) -> str:
        """Search the uploaded document to answer questions about its content."""
        context, is_relevant = get_relevant_context(query)
        if not is_relevant:
            return "No relevant information found in the document."
        return context

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a simple math expression, e.g. '5 + 5' or '12 * 3'."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    system_prompt = """You are a document Q&A assistant. Answer questions using ONLY
    the retrieved context you're given. If the context doesn't contain the answer,
    say "I don't know based on the provided document." Use the calculator tool only
    for math calculations."""

    llm = ChatGroq(model="openai/gpt-oss-120b")
    tools = [search_document, calculator]
    agent = create_agent(llm, tools, system_prompt=system_prompt)
    return agent, get_relevant_context


# --- File upload UI ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is None:
    st.info("👆 Upload a PDF to get started.")
    st.stop()  # halts execution here until a file is uploaded

# --- Save uploaded file to a temp location and hash it for caching ---
file_bytes = uploaded_file.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

temp_dir = tempfile.gettempdir()
temp_path = os.path.join(temp_dir, f"{file_hash}.pdf")
with open(temp_path, "wb") as f:
    f.write(file_bytes)

# --- Build (or reuse cached) agent for this specific file ---
with st.spinner("Processing document..."):
    agent, get_relevant_context = build_agent_for_file(file_hash, temp_path)

st.success(f"Ready! Ask questions about **{uploaded_file.name}**")

if st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

# --- Chat history, scoped per file so switching files resets the conversation ---
if "current_file_hash" not in st.session_state or st.session_state.current_file_hash != file_hash:
    st.session_state.current_file_hash = file_hash
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about the document...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context, is_relevant = get_relevant_context(user_input)

            if not is_relevant:
                answer = ("I don't know based on the provided document. "
                          "This question doesn't appear to be covered in the uploaded PDF.")
            else:
                augmented_input = f"""Answer the user's question using ONLY the context below.
If the context doesn't fully answer the question, say so.

Retrieved context from the document:
{context}

User's question: {user_input}"""

                result = agent.invoke({"messages": augmented_input})
                raw_answer = result["messages"][-1].content
                answer = re.sub(r'【.*?】', '', raw_answer).strip()

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})