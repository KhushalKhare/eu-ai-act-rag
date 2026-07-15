import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, local — good enough to prove the pipeline works
COLLECTION_NAME = "eu_ai_act"


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Fresh start each run — avoids stale/duplicate entries if you re-embed after edits
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "article_number": c["article_number"],
            "article_title": c["article_title"],
            "clause": c["clause"] or "",
            "source": c["source"],
        }
        for c in chunks
    ]

    print("Embedding chunks (this may take a minute on first run)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\nStored {collection.count()} chunks in Chroma at {DB_PATH}")

    # Sanity check: run one real query and see what comes back
    test_query = "What does the Act say about real-time biometric identification?"
    results = collection.query(query_texts=[test_query], n_results=3)
    print(f"\nTest query: '{test_query}'")
    for i, (cid, doc, meta) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0]
    )):
        print(f"\n{i+1}. [{cid}] Article {meta['article_number']}, clause {meta['clause']}")
        print(f"   {doc[:150]}...")


if __name__ == "__main__":
    main()