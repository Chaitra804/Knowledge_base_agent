# streamlit_app.py
import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv

# Google Gemini
import google.generativeai as genai

# LangChain components (community packages)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

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
# App UI config
# -------------------------------
st.set_page_config(page_title="Gemini Knowledge Base Agent", layout="wide", page_icon="📚")
st.title("📚 Gemini Knowledge Base Agent (Robust Retrieval + Hybrid Mode)")

st.markdown(
    "Upload PDF/TXT files, the app will index them and answer your questions. "
    "Use Hybrid Mode to allow the model to answer from general knowledge if documents lack the info."
)

# -------------------------------
# Session state
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
# Embeddings wrapper
# -------------------------------
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            r = genai.embed_content(model="models/text-embedding-004", content=t)
            # handle dict-like vs object
            emb = r.get("embedding") if isinstance(r, dict) else r["embedding"]
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        r = genai.embed_content(model="models/text-embedding-004", content=text)
        return r.get("embedding") if isinstance(r, dict) else r["embedding"]

# -------------------------------
# Helper functions
# -------------------------------
def load_and_split_documents(file_paths, chunk_size=1000, chunk_overlap=50):
    docs = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

def create_vectorstore(chunks, persist_dir="chroma_db", collection_name="gemini_collection"):
    embed = GeminiEmbeddings()
    texts = [c.page_content for c in chunks]

    # Remove existing persisted DB to avoid stale index (safe for demos)
    if os.path.exists(persist_dir):
        try:
            old = Chroma(persist_directory=persist_dir, collection_name=collection_name)
            try:
                old._client.close()
            except Exception:
                pass
            shutil.rmtree(persist_dir)
        except Exception:
            pass

    vectordb = Chroma.from_texts(texts=texts, embedding=embed, persist_directory=persist_dir, collection_name=collection_name)
    try:
        vectordb.persist()
    except Exception:
        # ephemeral filesystems (like Streamlit Cloud) may not persist
        pass

    # Return both vectordb and a retriever if possible
    retriever = None
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    except Exception:
        retriever = vectordb  # fallback, may offer similarity_search
    return vectordb, retriever

def robust_retrieve(retriever, vectordb, query, k):
    """
    Try multiple retrieval APIs in order:
      1. retriever.get_relevant_documents(query)
      2. retriever.retrieve(query)  (some libs)
      3. retriever.similarity_search(query, k)
      4. vectordb.similarity_search(query, k)
    Return list of documents (each with .page_content ideally).
    """
    results = []
    # 1 - LangChain Retriever
    try:
        if hasattr(retriever, "get_relevant_documents"):
            results = retriever.get_relevant_documents(query)
            return results
    except Exception:
        pass

    # 2 - generic retriever API
    try:
        if hasattr(retriever, "retrieve"):
            results = retriever.retrieve(query)
            return results
    except Exception:
        pass

    # 3 - similarity_search on retriever (some implementations)
    try:
        if hasattr(retriever, "similarity_search"):
            results = retriever.similarity_search(query, k=k)
            return results
    except Exception:
        pass

    # 4 - try vectordb directly
    try:
        if vectordb is not None and hasattr(vectordb, "similarity_search"):
            results = vectordb.similarity_search(query, k=k)
            return results
    except Exception:
        pass

    # 5 - try raw search / query on vectordb (best-effort)
    try:
        if vectordb is not None and hasattr(vectordb, "search"):
            results = vectordb.search(query, k=k)
            return results
    except Exception:
        pass

    return results

def generate_answer(model_name, prompt):
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    # robust extraction
    ans = None
    try:
        if hasattr(resp, "text") and resp.text:
            ans = resp.text
        elif isinstance(resp, dict):
            # some shapes
            if "candidates" in resp and len(resp["candidates"]) > 0:
                first = resp["candidates"][0]
                if isinstance(first, dict) and "content" in first:
                    ans = first["content"]
                else:
                    ans = str(first)
            elif "output" in resp:
                ans = str(resp["output"])
            else:
                ans = str(resp)
        else:
            ans = str(resp)
    except Exception as e:
        ans = f"Error parsing model response: {e}\nRaw response: {resp}"
    return ans

# -------------------------------
# Main app logic
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

    # Load & split
    with st.spinner("Processing documents and creating chunks..."):
        chunks = load_and_split_documents(file_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.success(f"✅ Created {len(chunks)} chunks.")
        st.session_state.chunks = chunks

    if show_debug:
        st.subheader("Debug — sample chunks (first 6)")
        for i, c in enumerate(chunks[:6]):
            st.markdown(f"**Chunk {i+1} (len={len(c.page_content)}):**")
            st.code(c.page_content[:1200])

    # Build vectorstore
    with st.spinner("Building vector store..."):
        vectordb, retriever = create_vectorstore(chunks)
        st.session_state.vectordb = vectordb
        st.session_state.retriever = retriever
    st.success("✅ Vector store ready.")

    # Query UI
    st.subheader("Ask a question")
    query = st.text_input("Enter your question (e.g., What is myopia?)")

    if query:
        with st.spinner("Retrieving relevant chunks..."):
            results = robust_retrieve(retriever, vectordb, query, k=top_k)

            # If nothing returned, try an alternate expanded query (helps short queries)
            if not results:
                alt_query = query if len(query) > 4 else f"{query} definition"
                try:
                    results = robust_retrieve(retriever, vectordb, alt_query, k=max(4, top_k))
                    if results:
                        st.info(f"Found {len(results)} chunks with alternate query '{alt_query}'.")
                except Exception:
                    pass

            # results should be list-like
            if not results:
                st.warning("No relevant chunks found in the indexed documents.")
            else:
                st.success(f"Retrieved {len(results)} chunk(s).")

        # Display retrieved chunks with optional toggle to show full text
        context_pieces = []
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

        # Build prompt: prefer doc context; hybrid_mode determines fallback behavior
        if hybrid_mode:
            prompt = f"""
You are an expert assistant. Use the CONTEXT to answer precisely. If the context doesn't contain the necessary info, answer from general knowledge and prefix the answer with "(Answer based on general knowledge)".

CONTEXT:
{context}

QUESTION:
{query}
"""
        else:
            prompt = f"""
You are an expert assistant. Use ONLY the following CONTEXT to answer. If the answer is not in the context, reply: "The document does not contain this information."

CONTEXT:
{context}

QUESTION:
{query}
"""

        # Send to Gemini and render answer
        try:
            answer = generate_answer(model_name="gemini-2.0-flash-lite", prompt=prompt)
        except Exception as e:
            answer = f"Error generating answer: {e}"

        # Save and show
        st.session_state.chat_history.append({"question": query, "answer": answer})
        st.markdown("### Answer")
        st.write(answer)

    # Show chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history[::-1]:
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.markdown("---")

else:
    st.info("Upload files to begin. After upload, ask questions about the uploaded documents.")
