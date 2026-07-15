import os
import json
from pathlib import Path
from typing import TypedDict, Optional
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from langgraph.graph import StateGraph, END

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "eu_ai_act"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = chroma_client.get_collection(COLLECTION_NAME)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class RAGState(TypedDict):
    original_query: str
    current_query: str
    retrieved_chunks: list
    iteration: int
    grade: Optional[str]
    answer: Optional[str]


def retrieve_node(state: RAGState) -> RAGState:
    query = state["current_query"]
    embedding = embed_model.encode([query]).tolist()

    results = collection.query(query_embeddings=embedding, n_results=3)

    chunks = []
    for cid, doc, meta in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    ):
        chunks.append({"chunk_id": cid, "text": doc, "metadata": meta})

    print(f"[retrieve] query='{query}' -> {len(chunks)} chunks")
    for c in chunks:
        print(f"   [{c['chunk_id']}] {c['text'][:80]}...")

    return {**state, "retrieved_chunks": chunks}


def grade_node(state: RAGState) -> RAGState:
    query = state["original_query"]
    chunks = state["retrieved_chunks"]

    chunks_text = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in chunks
    )

    prompt = f"""You are grading whether retrieved legal text actually answers a question about the EU AI Act.

Question: {query}

Retrieved chunks:
{chunks_text}

Do these chunks contain enough information to directly answer the question?
Respond with ONLY a JSON object, no other text: {{"grade": "pass" or "fail", "reasoning": "one short sentence"}}"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
        grade = result.get("grade", "fail")
        reasoning = result.get("reasoning", "")
    except json.JSONDecodeError:
        print(f"[grade] WARNING: couldn't parse LLM output as JSON: {raw!r}")
        grade = "fail"
        reasoning = "parse error, defaulting to fail-safe"

    print(f"[grade] {grade} — {reasoning}")

    return {**state, "grade": grade}


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")
graph.add_edge("grade", END)
app = graph.compile()


if __name__ == "__main__":
    test_query = "What does the Act say about tax obligations for AI companies?"
   # test_query = "What does the Act say about tax obligations for AI companies?"
    initial_state: RAGState = {
        "original_query": test_query,
        "current_query": test_query,
        "retrieved_chunks": [],
        "iteration": 0,
        "grade": None,
        "answer": None,
    }
    final_state = app.invoke(initial_state)
    print("\nFinal grade:", final_state["grade"])