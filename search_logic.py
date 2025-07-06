import os
import re
import requests
from typing import List, Dict
from dotenv import load_dotenv
import spacy
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

CREDIBLE_DOMAINS = [
    "bbc.com",
    "reuters.com",
    "ndtv.com",
    "indianexpress.com",
    "thehindu.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "thewire.in",
    "scroll.in",
    "economictimes.indiatimes.com",
    "moneycontrol.com",
    "livemint.com",
    "news18.com",
    "business-standard.com",
    "cnn.com",
    "nytimes.com",
    "aljazeera.com",
    "theguardian.com",
    "abcnews.go.com",
    "npr.org",
    "cbsnews.com"
]


SIMILARITY_THRESHOLD = 0.70
MAX_RELATED_ARTICLES = 5

def extract_named_entities(text: str, labels=("ORG", "PERSON", "GPE")) -> List[str]:
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ in labels))

def extract_keywords(text: str) -> List[str]:
    return extract_named_entities(text) or text.split()[:5]

def real_web_search(keywords: List[str], num_results: int = 10) -> List[Dict]:
    query = " ".join(keywords)
    url = (
        f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}"
        f"&cx={SEARCH_ENGINE_ID}&q={query}&num={num_results}"
    )
    try:
        response = requests.get(url)
        results = response.json()
        return [
            {
                "url": item["link"],
                "snippet": item.get("snippet", "").lower(),
                "title": item.get("title", "")
            }
            for item in results.get("items", [])
        ]
    except Exception as e:
        print("Search error:", e)
        return []

def is_credible_domain(url: str) -> bool:
    return any(domain in url for domain in CREDIBLE_DOMAINS)

def calculate_semantic_similarity(text: str, snippet: str) -> float:
    embeddings = embedder.encode([text, snippet], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

def score_news(text: str) -> dict:
    keywords = extract_keywords(text)
    print("🔍 Keywords:", keywords)

    results = real_web_search(keywords)
    print("📑 Search Results:", results)

    credible_and_similar = 0
    evidence = []
    related_articles = []

    for result in results:
        url = result["url"]
        snippet = result["snippet"]
        title = result["title"]
        credible = is_credible_domain(url)
        similarity = calculate_semantic_similarity(text.lower(), snippet)

        print(f"\n🧠 Comparing to snippet: {snippet}")
        print(f"→ Credible Domain: {credible}")
        print(f"→ Semantic Similarity: {similarity:.3f}")

        article_entry = {
            "url": url,
            "snippet": snippet,
            "title": title,
            "similarity": round(similarity, 3)
        }

        if credible and similarity >= SIMILARITY_THRESHOLD:
            evidence.append(article_entry)
            credible_and_similar += 1
            print("✅ Counted as credible + semantically matching.")
        else:
            related_articles.append(article_entry)
            print("❌ Not counted.")

    # Scoring logic
    score = {0: 0, 1: 50, 2: 70, 3: 85}.get(credible_and_similar, 100)
    is_real = score >= 70

    # Limit related articles to 5 max
    related_articles = sorted(related_articles, key=lambda x: -x["similarity"])[:MAX_RELATED_ARTICLES]

    return {
        "is_real": is_real,
        "confidence": score,
        "evidence": evidence,
        "related_articles": related_articles
    }

# Testing
if __name__ == "__main__":
    news = "Ameesha Patel is planning to marry Salman Khan and have good-looking babies together."
    print("📰 News:", news)
    result = score_news(news)
    print("🧾 Result:", result)
