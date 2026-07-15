import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

BASE_URL = "https://artificialintelligenceact.eu/article/{}/"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start narrow: Ch. I (1-4), Ch. II prohibited practices (5), Ch. III high-risk classification (6-7)
ARTICLE_RANGE = range(1, 8)

def fetch_article(article_num: int) -> dict | None:
    url = BASE_URL.format(article_num)
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        print(f"Article {article_num}: HTTP {resp.status_code}, skipping")
        return None
    
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")

    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else f"Article {article_num}"

    body_el = soup.find("div", class_="et_pb_post_content")
    if not body_el:
        print(f"Article {article_num}: couldn't find body content, check selector")
        return None

    text = body_el.get_text(separator="\n", strip=True)

    return {
        "article_number": article_num,
        "title": title,
        "text": text,
        "source": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ%3AL_202401689",
        "scraped_from": url,
    }

def main():
    articles = []
    for num in ARTICLE_RANGE:
        result = fetch_article(num)
        if result:
            articles.append(result)
            print(f"✓ Article {num}: {result['title']}")
        time.sleep(1)  # don't hammer the site

    out_path = OUTPUT_DIR / "articles.json"
    #out_path.write_text(json.dumps(articles, indent=2, ensure_ascii=False))
    out_path.write_text(json.dumps(articles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(articles)} articles to {out_path}")

if __name__ == "__main__":
    main()