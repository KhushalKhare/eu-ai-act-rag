import json
import re
from pathlib import Path

# Resolve paths relative to the project root, not wherever the script is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def has_numbered_clauses(text: str) -> bool:
    """Heuristic: does this article use (1), (2), (3)... structure (e.g. definitions)?
    Checked before lettered clauses, since numbered lists are the more specific/primary
    structure in articles like Article 3 (Definitions)."""
    return len(re.findall(r'\n\(\d+\)\s', text)) >= 3


def has_lettered_clauses(text: str) -> bool:
    """Heuristic: does this article have (a), (b), (c)... structure at the top level?"""
    return len(re.findall(r'\n\([a-z]\)\s', text)) >= 2


def split_by_numbered_clause(article: dict) -> list[dict]:
    """One chunk per numbered item, e.g. definitions (1), (2), (3)...
    Any (a)(b) sub-lettering *inside* a numbered item stays attached to that
    item's chunk — it's a sub-part of that definition, not a new top-level clause.
    This avoids the duplicate-ID bug where nested (a)(b) lists reset the letter
    counter and silently overwrite earlier same-named chunks."""
    text = article["text"]
    parts = re.split(r'\n(?=\(\d+\)\s)', text)

    chunks = []
    intro = parts[0].strip()
    if intro:
        chunks.append({
            "chunk_id": f"art{article['article_number']}_intro",
            "article_number": article["article_number"],
            "article_title": article["title"],
            "clause": None,
            "text": intro,
            "source": article["source"],
        })

    for part in parts[1:]:
        part = part.strip()
        match = re.match(r'\((\d+)\)', part)
        num = match.group(1) if match else None
        chunks.append({
            "chunk_id": f"art{article['article_number']}_def{num or 'x'}",
            "article_number": article["article_number"],
            "article_title": article["title"],
            "clause": num,
            "text": part,
            "source": article["source"],
        })
    return chunks


def split_by_subclause(article: dict) -> list[dict]:
    """One chunk per lettered clause — precise retrieval for structured articles
    like Article 5 (Prohibited AI Practices)."""
    text = article["text"]
    parts = re.split(r'\n(?=\([a-z]\)\s)', text)

    chunks = []
    intro = parts[0].strip()
    if intro:
        chunks.append({
            "chunk_id": f"art{article['article_number']}_intro",
            "article_number": article["article_number"],
            "article_title": article["title"],
            "clause": None,
            "text": intro,
            "source": article["source"],
        })

    for part in parts[1:]:
        part = part.strip()
        match = re.match(r'\(([a-z])\)', part)
        clause_letter = match.group(1) if match else None
        chunks.append({
            "chunk_id": f"art{article['article_number']}_{clause_letter or 'x'}",
            "article_number": article["article_number"],
            "article_title": article["title"],
            "clause": clause_letter,
            "text": part,
            "source": article["source"],
        })
    return chunks


def split_fixed(article: dict, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Fallback for unstructured prose: fixed-size sliding window."""
    text = article["text"]
    chunks = []
    start, idx = 0, 0
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append({
                "chunk_id": f"art{article['article_number']}_chunk{idx}",
                "article_number": article["article_number"],
                "article_title": article["title"],
                "clause": None,
                "text": piece,
                "source": article["source"],
            })
            idx += 1
        start += chunk_size - overlap
    return chunks


def dedupe_chunk_ids(chunks: list[dict]) -> list[dict]:
    """Safety net: if any chunk_id still collides (shouldn't happen after the
    numbered/lettered fix, but articles are messy legal text and edge cases exist),
    suffix it with a counter instead of silently overwriting data downstream."""
    seen = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in seen:
            seen[cid] += 1
            chunk["chunk_id"] = f"{cid}__dup{seen[cid]}"
        else:
            seen[cid] = 0
    return chunks


def main():
    articles = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    all_chunks = []

    for article in articles:
        num = article["article_number"]
        if has_numbered_clauses(article["text"]):
            chunks = split_by_numbered_clause(article)
            print(f"Article {num}: {len(chunks)} numbered-definition chunks")
        elif has_lettered_clauses(article["text"]):
            chunks = split_by_subclause(article)
            print(f"Article {num}: {len(chunks)} sub-clause chunks")
        else:
            chunks = split_fixed(article)
            print(f"Article {num}: {len(chunks)} fixed-size chunks")
        all_chunks.extend(chunks)

    all_chunks = dedupe_chunk_ids(all_chunks)

    ids = [c["chunk_id"] for c in all_chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids survived dedupe — investigate."

    OUTPUT_PATH.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(all_chunks)} total chunks to {OUTPUT_PATH}")
    print(f"Unique chunk_ids confirmed: {len(set(ids))}/{len(ids)}")


if __name__ == "__main__":
    main()