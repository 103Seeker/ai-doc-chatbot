# 🤖 AI Document Chatbot
**RAG + Multi-Agent AI system built with FastAPI, LangChain, LangGraph, ChromaDB & Groq**

---

## 🏗️ Architecture

```
User uploads PDF
    ↓
[Phase 1] FastAPI receives file → extracts text (pdfplumber)
    ↓
[Phase 2] Text → chunks → embeddings → ChromaDB (RAG)
    ↓
[Phase 3] User asks question → Multi-Agent pipeline:
    ├── 🔍 Retriever Agent  → finds relevant chunks
    ├── 🤖 Answer Agent     → generates answer via Groq LLM
    └── ✍️  Summarizer Agent → polishes final response
```

---

## 🚀 Local Setup

### 1. Clone & enter project
```bash
git clone <your-repo-url>
cd ai-doc-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
- Go to https://console.groq.com → create free account → copy API key
- Edit `.env` file:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Run the server
```bash
uvicorn main:app --reload
```

### 6. Open API docs
```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload-pdf` | Upload & index a PDF |
| GET | `/get-text/{doc_id}` | Get extracted text |
| GET | `/list-docs` | List all documents |
| DELETE | `/delete-doc/{doc_id}` | Delete a document |
| POST | `/search` | Semantic search only |
| POST | `/chat` | Full multi-agent RAG chat |

---

## 🧪 Test It

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Upload a PDF
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@your_document.pdf"

# 3. Chat with it
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "doc_id": "your_document"}'
```

---

## ☁️ Deploy to Railway (Free)

1. Push code to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Add environment variable: `GROQ_API_KEY=your_key`
4. Railway auto-detects FastAPI and deploys
5. Copy the public URL → add to your resume!

---

## 📁 Project Structure

```
ai-doc-chatbot/
├── main.py                  # FastAPI app, all routes
├── services/
│   ├── pdf_service.py       # PDF text extraction
│   ├── vector_service.py    # ChromaDB + embeddings (RAG)
│   └── agent_service.py     # LangGraph multi-agent pipeline
├── chroma_db/               # Auto-created, vector storage
├── requirements.txt
├── .env                     # Your API keys (never commit this!)
└── README.md
```

---

## 🛠️ Tech Stack

- **FastAPI** — Backend & REST APIs
- **pdfplumber** — PDF text extraction
- **LangChain** — RAG pipeline, text splitting
- **ChromaDB** — Vector database
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` (free, local)
- **LangGraph** — Multi-agent orchestration
- **Groq** — Free, fast LLM inference (LLaMA 3)
