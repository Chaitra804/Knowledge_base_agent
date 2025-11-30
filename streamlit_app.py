# streamlit_app.py
"""
Gemini Knowledge Base Agent - patched file
- Uses langchain_chroma.Chroma if available (recommended)
- Writes Chroma persistence to a writable temp dir to avoid read-only DB errors
- Falls back to TF-IDF retriever if Chroma or embeddings fail
- Guards vectordb.persist() calls (newer Chroma auto-persists)
"""

import os
import tempfile
import shutil
import traceback
import streamlit as st
from dotenv import load_dotenv

# Google Gemini
import google.generativeai as genai

# LangChain components (community packages)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Prefer the new langchain_chroma package if installed; otherwise fall back.
try:
    from langchain_chroma import Chroma
    CHROMA_IMPORT = "langchain_chroma"
except Exception:
    # fallback to older import (may be deprecated)
    try:
        from langchain_community.vectorstores import Chroma
        CHROMA_IMPORT = "langchain_community.vectorstores"
    except Exception:
        Chroma = None
        CHROMA_IMPORT = None

# Embeddings base
from langchain.embeddings.base import Embeddings

# TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -------------------------------
# Load API Key (from env or Streamlit secrets)
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None)

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY in .env or Streamlit Secrets. Please add it and restart.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Gemini Knowledge Base Agent", layout="wide", page_icon="📚")
st.title("📚 Gemini Knowledge Base Agent (Patched: langchain-chroma + safe persist)")
st.markdown(
    "Upload PDFs or TXT files and ask questions based on their content. "
    "This version prefers `langchain_chroma` for Chroma and writes persistence to a writable temp directory to avoid read-only DB errors."
)
st.caption(f"Chroma import used: {CHROMA_IMPORT}")

# -------------------------------
# Session state initialization
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Indexing & Retrieval Settings")
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=50, step=10)
top_k = st.sidebar.number_input("Top-k (retriever)", min_value=1, max_value=12, value=4, step=1)
hybrid_mode = st.sidebar.checkbox("Hybrid Mode (allow general-knowledge fallback)", value=True)
show_debug = st.sidebar.checkbox("Show debug chunk previews", value=False)
show_full_chunk_button = st.sidebar.checkbox("Show full chunk toggle", value=True)

# -------------------------------
# File upload
# -------------------------------
uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# -------------------------------
# Gemini Embeddings wrapper
# -------------------------------
class GeminiEmbeddings(Embeddings):
    """Wrapper for Gemini embeddings with LangChain-like interface."""
    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            r = genai.embed_content(model="models/text-embedding-004", content=t)
            if isinstance(r, dict):
                emb = r.get("embedding")
            else:
                emb = r["embedding"] if "embedding" in r else None
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        r = genai.embed_content(model="models/text-embedding-004", content=text)
        if isinstance(r, dict):
            return r.get("embedding")
        return r["embedding"] if "embedding" in r else None

# -------------------------------
# TF-IDF fallback retriever
# -------------------------------
class TfidfRetriever:
    """Simple TF-IDF retriever with LangChain-like interface."""
    def __init__(self, docs):
        safe_docs = [d if d and isinstance(d, str) else "" for d in docs]
        self.docs = safe_docs
        self.vectorizer = TfidfVectorizer().fit(safe_docs)
        self.doc_vectors = self.vectorizer.transform(safe_docs)

    def get_relevant_documents(self, query, k=4):
        qv = self.vectorizer.transform([query])
        scores = (self.doc_vectors * qv.T).toarray().squeeze()
        if scores.size == 0:
            return []
        idx = np.argsort(scores)[::-1][:k]
        results = []
        for i in idx:
            class D: pass
            d = D()
            d.page_content = self.docs[i]
            results.append(d)
        return results

    def similarity_search(self, query, k=4):
        return self.get_relevant_documents(query, k=k)

# -------------------------------
# Helper functions
# -------------------------------
def load_and_split_documents(file_paths, chunk_size=1000, chunk_overlap=50):
    """Load PDFs/TXT and split into chunks."""
    docs = []
    for path in file_paths:
        try:
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding="utf8")
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

def create_vectorstore(chunks, persist_dir="chroma_db", collection_name="gemini_collection", k=4):
    """
    Build Chroma vectorstore using a writable temp directory for persistence (to avoid readonly db).
    Falls back to TF-IDF retriever if anything fails.
    Returns either (vectordb, retriever) or a retriever instance (fallback).
    """
    embed = GeminiEmbeddings()
    documents = [chunk.page_content for chunk in chunks]

    if not documents:
        st.error("No documents found to index.")
        return None

    # Quick embedding validation
    try:
        test_emb = embed.embed_documents([documents[0]])
        if not test_emb or not isinstance(test_emb[0], (list, tuple)):
            st.warning("Embedding API returned unexpected result. Falling back to TF-IDF retriever.")
            return TfidfRetriever(documents)
        emb_len = len(test_emb[0])
        st.info(f"Sample embedding length: {emb_len}")
    except Exception as e:
        st.exception(e)
        st.warning("Embedding generation failed — falling back to TF-IDF retriever.")
        return TfidfRetriever(documents)

    # Create a writable temporary directory for Chroma persistence
    session_temp = None
    writable_persist_dir = None
    try:
        session_temp = tempfile.mkdtemp(prefix="chroma_")
        writable_persist_dir = os.path.join(session_temp, persist_dir)
        os.makedirs(writable_persist_dir, exist_ok=True)
    except Exception:
        writable_persist_dir = None

    # Try to create Chroma vectorstore
    try:
        if Chroma is None:
            st.warning("Chroma package not available; falling back to TF-IDF retriever.")
            return TfidfRetriever(documents)

        if writable_persist_dir:
            st.info(f"Using writable persist directory: {writable_persist_dir}")
            vectordb = Chroma.from_texts(
                texts=documents,
                embedding=embed,
                persist_directory=writable_persist_dir,
                collection_name=collection_name
            )
        else:
            st.info("Using in-memory Chroma (no persist_directory).")
            # Many Chroma versions accept no persist_directory for in-memory usage
            vectordb = Chroma.from_texts(
                texts=documents,
                embedding=embed,
                collection_name=collection_name
            )

        # Try to persist only if method exists; guard it
        try:
            if hasattr(vectordb, "persist"):
                try:
                    vectordb.persist()
                except Exception:
                    # ignore persistence errors; environment may be ephemeral
                    st.info("Chroma.persist() failed or not supported in this environment; continuing with in-memory index.")
        except Exception:
            pass

        # Return retriever
        try:
            retriever = vectordb.as_retriever(search_kwargs={"k": k})
            return vectordb, retriever
        except Exception:
            return vectordb, vectordb

    except Exception:
        # Capture traceback for debugging
        tb = traceback.format_exc()
        st.error("ChromaDB creation/upsert failed — falling back to TF-IDF retriever.")
        # Show truncated traceback in UI to help debug (avoid printing secrets)
        st.code("\n".join(tb.splitlines()[:300]))
        # Cleanup temp dir if it exists
        try:
            if session_temp and os.path.exists(session_temp):
                shutil.rmtree(session_temp)
        except Exception:
            pass
        return TfidfRetriever(documents)

def robust_retrieve(retriever_obj, vectordb_obj, query, k=4):
    """
    Try multiple retrieval APIs in order. Return list of Document-like objects.
    """
    results = []
    try:
        if retriever_obj and hasattr(retriever_obj, "get_relevant_documents"):
            return retriever_obj.get_relevant_documents(query)
    except Exception:
        pass
    try:
        if retriever_obj and hasattr(retriever_obj, "retrieve"):
            return retriever_obj.retrieve(query)
    except Exception:
        pass
    try:
        if retriever_obj and hasattr(retriever_obj, "similarity_search"):
            return retriever_obj.similarity_search(query, k=k)
    except Exception:
        pass
    try:
        if vectordb_obj and hasattr(vectordb_obj, "similarity_search"):
            return vectordb_obj.similarity_search(query, k=k)
    except Exception:
        pass
    return results

def generate_answer_with_gemini(model_name, prompt):
    """Call Gemini and extract a readable answer robustly."""
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    answer = None
    try:
        if hasattr(resp, "text") and resp.text:
            answer = resp.text
        elif isinstance(resp, dict):
            if "candidates" in resp and isinstance(resp["candidates"], list) and resp["candidates"]:
                first = resp["candidates"][0]
                if isinstance(first, dict) and "content" in first:
                    answer = first["content"]
                else:
                    answer = str(first)
            elif "output" in resp:
                answer = str(resp["output"])
            else:
                answer = str(resp)
        else:
            answer = str(resp)
    except Exception as e:
        answer = f"Error parsing model response: {e}\nRaw: {resp}"
    return answer

# -------------------------------
# Main App Logic
# -------------------------------
if uploaded_files:
    tmpdir = tempfile.mkdtemp()
    file_paths = []
    st.info("Saving uploaded files...")
    for uf in uploaded_files:
        p = os.path.join(tmpdir, uf.name)
        with open(p, "wb") as f:
            f.write(uf.read())
        file_paths.append(p)
        st.write(f"✅ Uploaded: {uf.name}")

    # Load & split documents
    with st.spinner("Processing documents and creating chunks..."):
        chunks = load_and_split_documents(file_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.success(f"✅ Created {len(chunks)} chunks.")
        st.session_state.chunks = chunks

    # Show chunk previews if requested
    if show_debug:
        st.subheader("Debug — sample chunk previews (first 6)")
        for i, c in enumerate(chunks[:6]):
            st.markdown(f"**Chunk {i+1} (length {len(c.page_content)}):**")
            st.code(c.page_content[:1200])

    # Build vectorstore (Chroma) with safe temp-dir persistence and fallback
    with st.spinner("Building vector store (Chroma) — this may take a moment..."):
        res = create_vectorstore(chunks, persist_dir="chroma_db", collection_name="gemini_collection", k=top_k)

        if isinstance(res, tuple) and len(res) == 2:
            vectordb, retriever = res
            st.session_state.vectordb = vectordb
            st.session_state.retriever = retriever
            st.success("✅ Chroma vector DB created and retriever ready.")
        elif res is None:
            st.error("Failed to create any retriever. Please check logs.")
            st.stop()
        else:
            # TF-IDF fallback retriever
            retriever = res
            vectordb = None
            st.session_state.retriever = retriever
            st.session_state.vectordb = None
            st.warning("Using TF-IDF fallback retriever (Chroma unavailable).")

    # Model selection
    model_choice = st.selectbox(
        "Choose a Gemini model",
        ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-pro"],
        index=0
    )

    # Query UI
    st.subheader("💬 Ask a Question or Summarize")
    query = st.text_input("Enter your question (or type 'summarize' for a full summary):")

    if query:
        with st.spinner("Retrieving relevant document chunks..."):
            results = robust_retrieve(st.session_state.retriever, st.session_state.vectordb, query, k=top_k)

            # If nothing returned, try an alternate expanded query
            if not results:
                st.warning("No relevant chunks found with the default query.")
                alt_query = query if len(query) > 4 else f"{query} definition"
                try:
                    results = robust_retrieve(st.session_state.retriever, st.session_state.vectordb, alt_query, k=max(4, top_k))
                    if results:
                        st.info(f"Found {len(results)} chunks with alternate query '{alt_query}'.")
                except Exception:
                    pass

            if not results:
                st.warning("No relevant chunks returned from retriever. The app may use Hybrid Mode (if enabled) to answer from general knowledge.")
            else:
                st.success(f"Retrieved {len(results)} chunk(s).")

        # Display retrieved chunks with optional expanders
        context_pieces = []
        if results:
            st.subheader("Retrieved Chunks (sources)")
            for i, r in enumerate(results[:top_k]):
                content = getattr(r, "page_content", None) or (r.get("page_content") if isinstance(r, dict) else str(r))
                snippet = content[:800]
                st.markdown(f"**Source {i+1} (len {len(content)}):**")
                st.code(snippet)
                if show_full_chunk_button:
                    with st.expander(f"Show full chunk {i+1}"):
                        st.write(content)
                context_pieces.append(content)

        context = "\n\n".join(context_pieces)

        # Build prompt
        if hybrid_mode:
            prompt = f"""
You are an expert assistant. Use the CONTEXT to answer the user's question whenever possible.
If the context doesn't contain the necessary info, answer from your general knowledge and prefix the answer with "(Answer based on general knowledge)".

CONTEXT:
{context}

QUESTION:
{query}
"""
        else:
            prompt = f"""
You are an expert assistant. Use ONLY the following CONTEXT to answer the user's question.
If the answer is not in the context, reply: "The document does not contain this information."

CONTEXT:
{context}

QUESTION:
{query}
"""

        # Call Gemini LLM
        try:
            with st.spinner(f"Generating answer using {model_choice}..."):
                answer = generate_answer_with_gemini(model_choice, prompt)
        except Exception as e:
            answer = f"Error generating answer: {e}"

        st.session_state.chat_history.append({"question": query, "answer": answer})

    # -------------------------------
    # Display Chat History with colored bubbles
    # -------------------------------
    if st.session_state.chat_history:
        st.subheader("🗒 Chat History")
        for chat in st.session_state.chat_history[::-1]:
            st.markdown(
                f"<div style='background-color:#E0F7FA;padding:10px;border-radius:10px;margin-bottom:5px;'>"
                f"<strong>Q:</strong> {chat['question']}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='background-color:#FFF9C4;padding:10px;border-radius:10px;margin-bottom:15px;'>"
                f"<strong>A:</strong> {chat['answer']}</div>",
                unsafe_allow_html=True
            )

else:
    st.info("Upload files to begin. After upload, ask questions about the uploaded documents.")
