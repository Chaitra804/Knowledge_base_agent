import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv

# Google Gemini
import google.generativeai as genai

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

# -------------------------------
# Load API Key
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY in .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(
    page_title="Gemini Knowledge Base Agent",
    layout="wide",
    page_icon="📚"
)
st.title("📚 Gemini Knowledge Base Agent")
st.markdown("""
Upload PDFs or TXT files and ask questions based on their content.  
Type `summarize` to get a summary of the uploaded documents.
""")

# -------------------------------
# Chat History
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# File Upload
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# -------------------------------
# Gemini Embeddings wrapper
# -------------------------------
class GeminiEmbeddings(Embeddings):
    """Wrapper for Gemini embeddings with LangChain interface."""

    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            result = genai.embed_content(model="models/text-embedding-004", content=t)
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, text):
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        return result["embedding"]

# -------------------------------
# Helper Functions
# -------------------------------
def load_and_split_documents(file_paths):
    """Load PDFs/TXT and split into chunks."""
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks

def create_vectorstore(chunks, persist_dir="chroma_db", collection_name="gemini_collection"):
    """Build Chroma vectorstore from document chunks using Gemini embeddings."""
    embed = GeminiEmbeddings()
    documents = [chunk.page_content for chunk in chunks]

    # Close old DB safely
    if os.path.exists(persist_dir):
        try:
            old_vectordb = Chroma(persist_directory=persist_dir, collection_name=collection_name)
            old_vectordb._client.close()
            shutil.rmtree(persist_dir)
        except Exception:
            st.warning("Old Chroma DB in use; continuing without deletion.")

    vectordb = Chroma.from_texts(
        texts=documents,
        embedding=embed,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    vectordb.persist()
    return vectordb.as_retriever(search_kwargs={"k": 4})

# -------------------------------
# Main App Logic
# -------------------------------
if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    # Save uploaded files
    st.info("Saving uploaded files...")
    for uf in uploaded_files:
        path = os.path.join(temp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.read())
        file_paths.append(path)
        st.write(f"✅ Uploaded: {uf.name}")

    # Load & split documents
    with st.spinner("Processing documents..."):
        chunks = load_and_split_documents(file_paths)
        st.success(f"✅ Total chunks created: {len(chunks)}")

    # Create vectorstore
    retriever = create_vectorstore(chunks)
    st.success("✅ Vector database created!")

    # Model selection
    model_choice = st.selectbox(
        "Choose a Gemini model",
        ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-pro"]
    )
    model_chat = genai.GenerativeModel(model_choice)

    # -------------------------------
    # Ask Question / Summarize UI
    # -------------------------------
    st.subheader("💬 Ask a Question or Summarize")
    query = st.text_input("Enter your question (or type 'summarize' for full summary):")

    if query:
        with st.spinner("Retrieving relevant document chunks..."):
            # Use invoke() for VectorStoreRetriever
            results = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in results])
            st.write(f"Retrieved {len(results)} chunks.")

        prompt = f"""
You are a helpful AI assistant.

Use ONLY the following context to answer the user's question.
If the answer is not in the context, respond:
"The document does not contain this information."

CONTEXT:
{context}

QUESTION:
{query}
"""

        try:
            with st.spinner(f"Generating answer using {model_choice}..."):
                response = model_chat.generate_content(prompt)
                answer = response.text
        except Exception as e:
            answer = f"Error generating answer: {e}"

        st.session_state.chat_history.append({"question": query, "answer": answer})

    # -------------------------------
    # Display Chat History with Colored Bubbles
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
    st.info("Upload files to begin.")
