import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Reuse the actual compiled LangGraph pipeline — this API is a thin wrapper,
# never a reimplementation. If graph.py changes, this automatically reflects it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import app as rag_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="EU AI Act RAG API")

# Local dev: allow the frontend to call this from any origin.
# Before deploying publicly, restrict allow_origins to your actual frontend domain —
# wide-open CORS is fine on localhost, not fine once this is reachable from the internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    article_number: int
    clause: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grade: str
    rewrites: int


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    initial_state = {
        "original_query": request.question,
        "current_query": request.question,
        "retrieved_chunks": [],
        "iteration": 0,
        "grade": None,
        "grade_reasoning": None,
        "answer": None,
    }

    final_state = rag_graph.invoke(initial_state)

    citations = [
        Citation(
            chunk_id=c["chunk_id"],
            article_number=c["metadata"]["article_number"],
            clause=c["metadata"]["clause"] or None,
        )
        for c in final_state["retrieved_chunks"]
    ]

    return ChatResponse(
        answer=final_state["answer"],
        citations=citations,
        grade=final_state["grade"],
        rewrites=final_state["iteration"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the frontend at the root URL, so opening http://localhost:8000 just works
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))