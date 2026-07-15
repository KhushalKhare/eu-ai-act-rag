import json
data = json.load(open("data/processed/articles.json", encoding="utf-8"))
art5 = next(a for a in data if a["article_number"] == 5)
print(repr(art5["text"][:200]))