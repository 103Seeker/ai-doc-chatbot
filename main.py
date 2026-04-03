from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from services.pdf_service import extract_text_from_pdf
from services.vector_service import store_chunks, search_chunks, clear_store
from services.agent_service import run_agent

app = FastAPI(
    title="AI Document Chatbot",
    description="RAG-powered multi-agent document chatbot built with FastAPI + LangChain + Groq",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_docs: dict[str, str] = {}


class ChatRequest(BaseModel):
    question: str
    doc_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    agent_trace: list[str]


# ─────────────────────────────────────────
# PHASE 1 — Core APIs
# ─────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "message": "AI Document Chatbot is running 🚀"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    text = extract_text_from_pdf(contents)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    doc_id = file.filename.replace(" ", "_").replace(".pdf", "")
    uploaded_docs[doc_id] = text

    num_chunks = store_chunks(doc_id, text)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "characters": len(text),
        "chunks_stored": num_chunks,
        "message": "PDF uploaded and indexed successfully ✅"
    }


@app.get("/get-text/{doc_id}")
def get_text(doc_id: str):
    if doc_id not in uploaded_docs:
        raise HTTPException(status_code=404, detail="Document not found.")
    text = uploaded_docs[doc_id]
    return {
        "doc_id": doc_id,
        "characters": len(text),
        "preview": text[:500] + ("..." if len(text) > 500 else ""),
        "full_text": text
    }


@app.get("/list-docs")
def list_docs():
    return {
        "documents": list(uploaded_docs.keys()),
        "count": len(uploaded_docs)
    }


@app.delete("/delete-doc/{doc_id}")
def delete_doc(doc_id: str):
    if doc_id not in uploaded_docs:
        raise HTTPException(status_code=404, detail="Document not found.")
    uploaded_docs.pop(doc_id)
    clear_store(doc_id)
    return {"message": f"Document '{doc_id}' deleted."}


# ─────────────────────────────────────────
# PHASE 2 — RAG Search
# ─────────────────────────────────────────

@app.post("/search")
def search(doc_id: str, query: str, top_k: int = 3):
    if doc_id not in uploaded_docs:
        raise HTTPException(status_code=404, detail="Document not found.")
    results = search_chunks(doc_id, query, top_k)
    return {"doc_id": doc_id, "query": query, "results": results}


# ─────────────────────────────────────────
# PHASE 3 — Multi-Agent Chat
# ─────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.doc_id not in uploaded_docs:
        raise HTTPException(status_code=404, detail="Document not found. Upload a PDF first.")

    result = run_agent(req.doc_id, req.question)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        agent_trace=result["trace"]
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)