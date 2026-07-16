import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import app, groq_client, GROQ_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_PATH = PROJECT_ROOT / "eval" / "qa_set.jsonl"
RESULTS_PATH = PROJECT_ROOT / "eval" / "results.json"


def load_qa_set() -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline(question: str) -> dict:
    initial_state = {
        "original_query": question,
        "current_query": question,
        "retrieved_chunks": [],
        "iteration": 0,
        "grade": None,
        "grade_reasoning": None,
        "answer": None,
    }
    final_state = app.invoke(initial_state)
    return {
        "answer": final_state["answer"],
        "contexts": [c["text"] for c in final_state["retrieved_chunks"]],
        "iterations": final_state["iteration"],
    }


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """Does the answer only claim things actually supported by the retrieved text?"""
    context_text = "\n\n".join(contexts)
    prompt = f"""Rate whether this answer is fully supported by the retrieved context, on a scale of 0.0 to 1.0.
1.0 = every claim in the answer is directly backed by the context.
0.0 = the answer contains claims not found in or contradicted by the context.

Context:
{context_text}

Answer:
{answer}

Respond with ONLY a JSON object: {{"score": 0.0-1.0, "reasoning": "one short sentence"}}"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return {"score": float(result.get("score", 0.0)), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": 0.0, "reasoning": "parse error"}


def score_relevancy(question: str, answer: str) -> dict:
    """Does the answer actually address what was asked?"""
    prompt = f"""Rate how well this answer addresses the question, on a scale of 0.0 to 1.0.
1.0 = directly and completely answers the question (including honestly saying information isn't available, if true).
0.0 = irrelevant, evasive without reason, or off-topic.

Question: {question}

Answer: {answer}

Respond with ONLY a JSON object: {{"score": 0.0-1.0, "reasoning": "one short sentence"}}"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return {"score": float(result.get("score", 0.0)), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": 0.0, "reasoning": "parse error"}


def main():
    qa_pairs = load_qa_set()
    print(f"Running {len(qa_pairs)} eval questions through the full pipeline...\n")

    records = []
    for i, pair in enumerate(qa_pairs, 1):
        print(f"[{i}/{len(qa_pairs)}] ({pair.get('category','?')}/{pair.get('difficulty','?')}) {pair['question']}")
        result = run_pipeline(pair["question"])
        faith = score_faithfulness(result["answer"], result["contexts"])
        rel = score_relevancy(pair["question"], result["answer"])
        print(f"    faithfulness={faith['score']:.2f}  relevancy={rel['score']:.2f}  rewrites={result['iterations']}")

        records.append({
            "id": pair.get("id"),
            "category": pair.get("category", "uncategorized"),
            "difficulty": pair.get("difficulty", "unspecified"),
            "question": pair["question"],
            "answer": result["answer"],
            "faithfulness": faith["score"],
            "relevancy": rel["score"],
            "rewrites": result["iterations"],
        })

    print("\n--- Per-question ---")
    for r in records:
        print(f"[{r['id']}] {r['category']}/{r['difficulty']}  faith={r['faithfulness']:.2f}  rel={r['relevancy']:.2f}")

    def avg_by(key):
        groups = {}
        for r in records:
            groups.setdefault(r[key], []).append(r)
        return {
            k: {
                "faithfulness": mean(x["faithfulness"] for x in v),
                "relevancy": mean(x["relevancy"] for x in v),
                "n": len(v),
            }
            for k, v in groups.items()
        }

    by_category = avg_by("category")
    by_difficulty = avg_by("difficulty")

    print("\n--- By category ---")
    for cat, scores in by_category.items():
        print(f"  {cat}: faithfulness={scores['faithfulness']:.2f} relevancy={scores['relevancy']:.2f} (n={scores['n']})")

    print("\n--- By difficulty ---")
    for diff, scores in by_difficulty.items():
        print(f"  {diff}: faithfulness={scores['faithfulness']:.2f} relevancy={scores['relevancy']:.2f} (n={scores['n']})")

    overall = {
        "faithfulness_avg": mean(r["faithfulness"] for r in records),
        "relevancy_avg": mean(r["relevancy"] for r in records),
        "n_questions": len(records),
    }
    print(f"\nOverall: {overall}")

    RESULTS_PATH.write_text(
        json.dumps({"overall": overall, "by_category": by_category, "by_difficulty": by_difficulty, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()