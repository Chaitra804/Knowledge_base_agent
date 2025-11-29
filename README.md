📚 Gemini Knowledge Base Agent
A Streamlit-powered Knowledge Base Agent that uses Google Gemini, ChromaDB, and LangChain to extract information from PDF/TXT documents and answer user queries intelligently using Retrieval-Augmented Generation (RAG).
This project allows users to upload documents, build a vector database, and ask questions whose answers are grounded in the content of those documents — with optional fallback to general AI reasoning.

🧠 Overview
The Gemini Knowledge Base Agent is an AI assistant designed to:
Read, index, and understand PDF & text files
Split documents into semantic chunks
Convert chunks into vector embeddings
Search for relevant chunks using similarity search
Answer user questions based ONLY on document context (Strict RAG)
OR combine document knowledge with AI reasoning (Hybrid Mode)
Provide chunk previews to ensure transparency and explainability

This makes it useful for:
Research & academic study
Document-based Q&A
Summarization
Policy or legal document review
Knowledge extraction
Technical documentation assistance

🚀 Features
✔ Document Upload
Upload multiple PDFs or text files and build an internal knowledge base on the fly.

✔ Chunk Splitting
Uses RecursiveCharacterTextSplitter for optimal chunk size and overlap.

✔ Embeddings with Gemini
Uses Google Gemini Text Embedding 004 for highly accurate vector embeddings.

✔ ChromaDB Vector Store
Fast, local vector database for similarity search.

✔ Smart Retrieval
Robust retrieval logic:
get_relevant_documents
similarity_search
Automatic fallback queries

✔ Hybrid Mode
If enabled:
Uses documents first
Falls back to Gemini’s reasoning if no chunk found

If disabled:
Pure RAG
Only answers from documents

✔ Debug Chunk Viewer
Preview chunks to confirm that important keywords (like myopia) were indexed.

✔ Chat History
Full Q&A history displayed in the UI.

✔ Streamlit UI
Clean, interactive, minimal UI with sidebar controls.

⚠️ Limitations
Even though the system is powerful, it has some limitations:

❌ Document persistence
ChromaDB running on Streamlit Cloud cannot persist across restarts.

❌ Embedding cost
Gemini embedding API calls consume credits.

❌ Large documents
Extremely large PDFs may cause slow processing.

❌ No OCR
Scanned PDFs (images) are not readable unless text is extractable.

❌ Single-user session
Streamlit does not preserve state across users.

🧠 Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	Python
LLM Model	Google Gemini 2.0 Flash Lite / Flash / 1.5 Pro
Embeddings	Gemini Text Embedding 004
Vector Store	ChromaDB
Text Splitting	LangChain Text Splitters
Document Loaders	PyPDFLoader, TextLoader
Environment Handling	python-dotenv

🔑 API Used
Google Gemini API

Used for:
Text Generation
Retrieval-Augmented Question Answering
Embedding generation for vector search

Models Used:
models/text-embedding-004
gemini-2.0-flash-lite
gemini-2.0-flash
gemini-1.5-pro

🛠️ Setup & Run Instructions (Locally)
1️⃣ Clone the repository
git clone https://github.com/Chaitra804/Knowledge_base_agent.git
cd Knowledge_base_agent

2️⃣ Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your Gemini API key
Create a file .env:
GEMINI_API_KEY=your_api_key_here

5️⃣ Run the app
streamlit run streamlit_app.py

Open in browser:
http://localhost:8501


🌐 Live Demo
👉 https://knowledgebaseagent-ah4xuswuvh37rc8h7u2uau.streamlit.app

🎯 Potential Improvements
Here are upgrades you can add in the future:
🚀 1. Cloud Vector Database
Use:
Pinecone
Weaviate
Qdrant Cloud
So your embeddings persist across sessions.

🚀 2. Full Document Summaries
Add:
chapter-level summary
auto-topic extraction
PDF metadata extraction

🚀 3. Multi-user Sessions
Store chat history in a database.

🚀 4. Better UI
Add:
collapsible chunk sources
token usage
performance analytics

🚀 5. OCR Support
Use:
Tesseract
PyMuPDF
For scanned PDFs.

🚀 6. File Type Expansion
Add support for:
DOCX
PPTX
HTML
Websites

🚀 7. API Backend
Expose the agent via a FastAPI or Flask endpoint.

🏗️ Project Architecture
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/7316f517-2256-4450-8411-b6a2a634f28a" />

DEMO VIDEO
