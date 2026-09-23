# 📘 PDF Q&A Assistant — RAG + Agent Pipeline

A document Q&A app that lets you upload any PDF and ask questions grounded strictly in its content. Built with LangChain, a local embedding model, Chroma, Groq, and a Streamlit UI.

**The problem this solves:** general-purpose LLMs don't know the contents of your private documents, and even when you paste a document into a prompt, long documents don't fit in a context window cheaply. Retrieval-Augmented Generation (RAG) solves this by searching the document for only the relevant chunks and grounding the model's answer in those chunks — preventing hallucination on domain-specific content.

## Demo

Upload a PDF → ask a question → get an answer grounded in the document, with page citations. Ask something outside the document's scope, and it explicitly says so instead of guessing.

## Architecture

```
PDF Upload (Streamlit file_uploader)
  → PyPDFLoader                         (extracts text, 1 Document per page)
  → RecursiveCharacterTextSplitter      (chunks ~500 chars, 50 char overlap)
  → HuggingFace all-MiniLM-L6-v2        (local embeddings, 384-dim vectors)
  → Chroma vector store                 (persisted per-file, keyed by content hash)
  → similarity_search_with_score        (retrieval with a measured relevance gate)
  → LangChain agent (create_agent)      (search_document + calculator tools)
  → Streamlit chat UI                   (per-file session history, source citations)
```

## Key design decisions

- **Local embeddings (HuggingFace), cloud inference (Groq):** Groq doesn't offer an embeddings API, so embeddings run locally and free via `sentence-transformers`, while generation uses Groq's fast, free-tier LLM inference.
- **Chroma over alternatives (Weaviate, Pinecone, Qdrant):** for a project at this scale, Chroma needs zero infrastructure — no server, no account, just a local folder. A production system serving many concurrent users would justify a managed/server-based vector DB instead.
- **Per-file vector stores, keyed by content hash (not filename):** re-uploading the same file reuses its cached embeddings; a different file — even with the same name — gets its own fresh store. Prevents cache collisions and avoids re-embedding unchanged files.
- **Grounding enforced in code, not just in the prompt.** Early versions relied on a system prompt instructing the agent to "always search the document first." This failed under testing — the model would still answer confidently from its own training knowledge for questions it was confident about (e.g., general SQL syntax), even when explicitly told not to. **Fix:** retrieval and a relevance check now run in plain Python *before* the LLM is ever called. If the best-matching chunk's similarity score falls outside a measured threshold, the LLM is never invoked for that turn — grounding is guaranteed structurally, not requested via prompting.
- **Threshold set empirically, not guessed.** Distance scores were logged across known-relevant and known-irrelevant queries to find an actual separating value, rather than picking an arbitrary number.

## Bugs found and fixed during development

| Bug | Fix |
|---|---|
| Vector store silently duplicated on every re-run of the build script | Check `os.path.exists(persist_dir)` before rebuilding; load instead of re-adding |
| Agent skipped document retrieval for questions it was "confident" about, hallucinating instead of grounding | Moved retrieval + relevance-gating out of the LLM's control into deterministic Python code, run before the agent is invoked |
| Groq model `llama-3.3-70b-versatile` deprecated mid-project | Migrated to `openai/gpt-oss-120b` |
| `create_react_agent` deprecated in LangGraph v1 | Migrated to `langchain.agents.create_agent`, including its renamed `system_prompt` parameter |
| Compiled dependency (`uuid_utils`) blocked by Windows Smart App Control | Diagnosed via `uv pip uninstall` + import testing; resolved via forced reinstall |
| Citation marker artifacts (`【...】`) leaking into displayed answers | Stripped via regex before rendering in the UI |

## Setup

```bash
# clone and enter the project
cd your-project-folder

# create and activate a virtual environment
uv venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# install dependencies
uv add streamlit langchain langchain-community langchain-groq langchain-huggingface langchain-chroma langchain-text-splitters langchain langgraph pypdf sentence-transformers python-dotenv chromadb

# add your Groq API key
echo "GROQ_API_KEY=your-key-here" > .env

# run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Tech stack

- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Vector store:** Chroma (local, persisted)
- **Orchestration:** LangChain + LangGraph (`create_agent`)
- **UI:** Streamlit

## Known limitations

- Relevance threshold (`1.15`) was tuned on one document's embedding distribution; a very different document/domain might need recalibration.
- Meta-questions about the system itself (e.g., "what document is this?") are treated as out-of-scope rather than answered directly, since they don't semantically match document content.
- Scanned/image-only PDFs aren't supported — `PyPDFLoader` requires text-based PDFs; OCR would need to be added as a preprocessing step for scanned documents.