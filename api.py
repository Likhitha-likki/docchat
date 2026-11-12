"""FastAPI backend.  Run with:  uvicorn api:app --reload   (docs at http://localhost:8000/docs)"""
import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docchat.pipeline import DocChat, cited_hits

app = FastAPI(title="DocChat API")
chat = DocChat()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@app.post("/ingest")
def ingest(file: UploadFile):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    try:
        added = chat.ingest(file.file.read(), file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"document": file.filename, "chunks_added": added}


@app.post("/ask")
def ask(req: AskRequest):
    if not chat.store.documents():
        raise HTTPException(409, "No documents indexed yet. POST a PDF to /ingest first.")
    hits, stream = chat.ask(req.question, req.top_k)
    try:
        answer = "".join(stream)
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e}")
    return {
        "answer": answer,
        "sources": [
            {"n": n, "document": h.chunk.doc, "page": h.chunk.page, "score": round(h.score, 3), "text": h.chunk.text}
            for n, h in cited_hits(answer, hits)
        ],
    }


@app.get("/documents")
def documents():
    return chat.store.documents()


@app.delete("/reset")
def reset():
    chat.store.reset()
    return {"status": "cleared"}
