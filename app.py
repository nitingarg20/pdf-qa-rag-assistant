import streamlit as st
import os
import tempfile
import hashlib
import re
import ast
import operator
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
    RELEVANCE_THRESHOLD = 1.15  # lower distance = more similar; tune based on testing
 
    def get_relevant_context(query: str):
        """Returns context, relevance status, and source page numbers."""

        results = vectorstore.similarity_search_with_score(query, k=3)

        if not results:
            return "", False, []

        # Lower distance means more similar
        best_distance = results[0][1]
        is_relevant = best_distance < RELEVANCE_THRESHOLD

        context = ""
        source_pages = []

        for doc, distance in results:

            # Extract page number from PDF metadata
            page = doc.metadata.get("page_label")

            if page is None:
                page = doc.metadata.get("page")

                if page is not None:
                    page = int(page) + 1

            if page is None:
                page = "Unknown"

            context += f"[Page {page}]: {doc.page_content}\n\n"

            if page != "Unknown" and page not in source_pages:
                source_pages.append(page)

        return context, is_relevant, source_pages   
 
    @tool
    def search_document(query: str) -> str:
        """Search the uploaded document to answer questions about its content."""
        context, is_relevant, source_pages = get_relevant_context(query)
        if not is_relevant:
            return "No relevant information found in the document."
        return context
 
    def safe_calculate(expression):
        operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value

            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                left = evaluate(node.left)
                right = evaluate(node.right)

                if isinstance(node.op, ast.Pow) and abs(right) > 100:
                    raise ValueError("Exponent too large")

                result = operations[type(node.op)](left, right)

                if isinstance(result, (int, float)) and abs(result) > 1e100:
                    raise ValueError("Result too large")

                return result

            if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
                return operations[type(node.op)](evaluate(node.operand))

            raise ValueError("Invalid expression")

        tree = ast.parse(expression, mode="eval")
        return evaluate(tree.body)


    @tool
    def calculator(expression: str) -> str:
        """Safely evaluate basic mathematical expressions."""
        try:
            return str(safe_calculate(expression))
        except Exception:
            return "Invalid mathematical expression"
 
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

        # Restore citations from previous answers
        if msg["role"] == "assistant" and msg.get("source_pages"):
            with st.expander("📄 View Sources"):
                st.write("Retrieved from PDF pages:")

                for page in msg["source_pages"]:
                    st.write(f"- Page {page}")
 
user_input = st.chat_input("Ask a question about the document...")
 


if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Initialize variables for every new question
    source_pages = []
    is_relevant = False
    is_calculation = False

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # 1. Detect calculation requests
            has_calculation_keyword = (
                re.search(
                    r"\b(calculate|compute|solve)\b",
                    user_input,
                    re.IGNORECASE
                )
                and re.search(r"\d", user_input)
            )

            has_math_expression = re.search(
                r"(?<![\w.])\d+(?:\.\d+)?\s*[+\-*/%]\s*\d+(?:\.\d+)?(?![\w.])",
                user_input
            )

            is_calculation = bool(
                has_calculation_keyword or has_math_expression
            )

            # 2. Calculation processing
            if is_calculation:

                result = agent.invoke({
                    "messages": f"""
                    The user is asking for a mathematical calculation.
                    You MUST use the calculator tool to calculate the result.
                    Do not search the uploaded document.

                    User question: {user_input}
                    """
                })

                raw_answer = result["messages"][-1].content
                answer = re.sub(r'【.*?】', '', raw_answer).strip()

            # 3. Document question processing
            else:

                context, is_relevant, source_pages = (
                    get_relevant_context(user_input)
                )

                if not is_relevant:
                    answer = (
                        "I don't know based on the provided document. "
                        "This question doesn't appear to be covered "
                        "in the uploaded PDF."
                    )

                else:
                    augmented_input = f"""
                    Answer the user's question using ONLY the context below.
                    If the context doesn't fully answer the question, say so.

                    Retrieved context from the document:
                    {context}

                    User's question: {user_input}
                    """

                    result = agent.invoke({
                        "messages": augmented_input
                    })

                    raw_answer = result["messages"][-1].content
                    answer = re.sub(
                        r'【.*?】', '', raw_answer
                    ).strip()

        # 4. Display the answer
        st.markdown(answer)

        # 5. Display source pages for document-based answers
        if not is_calculation and is_relevant and source_pages:
            with st.expander("📄 View Sources"):
                st.write("Retrieved from PDF pages:")

                for page in source_pages:
                    st.write(f"- Page {page}")

    # 6. Save assistant response for ALL types of questions
    st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "source_pages": source_pages if is_relevant and not is_calculation else []
})
