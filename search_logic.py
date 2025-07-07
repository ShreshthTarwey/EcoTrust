import os
import re
import requests
from typing import List, Dict
from dotenv import load_dotenv
import spacy
from sentence_transformers import SentenceTransformer, util
# Add KeyBERT for better keyword extraction
from keybert import KeyBERT
# Use a stronger embedding model
# from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-mpnet-base-v2")
kw_model = KeyBERT()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Add this to .env if using SerpAPI

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

SIMILARITY_THRESHOLD = 0.60
MAX_RELATED_ARTICLES = 5

# Improved keyword extraction using KeyBERT and SpaCy

def extract_keywords(text: str) -> List[str]:
    # Use KeyBERT for context-aware keywords
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    if keywords:
        return [kw[0] for kw in keywords]
    # Fallback to SpaCy NER
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PERSON", "GPE", "EVENT")]
    return ents or text.split()[:5]

# Use SerpAPI for fresher results (fallback to Google Custom Search if needed)
def real_web_search(keywords: List[str], num_results: int = 10, days: int = 7) -> List[Dict]:
    query = " ".join(keywords)
    serpapi_url = f"https://serpapi.com/search.json?q={query}&engine=google&api_key={SERPAPI_KEY}&num={num_results}&tbs=qdr:d{days}"
    try:
        response = requests.get(serpapi_url)
        results = response.json()
        if "organic_results" in results:
            return [
                {
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "").lower(),
                    "title": item.get("title", ""),
                    "date": item.get("date", "")
                }
                for item in results["organic_results"]
            ]
    except Exception as e:
        print("SerpAPI search error:", e)
    # Fallback to Google Custom Search
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
                "title": item.get("title", ""),
                "date": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time", "")
            }
            for item in results.get("items", [])
        ]
    except Exception as e:
        print("Google search error:", e)
        return []

def is_credible_domain(url: str) -> bool:
    return any(domain in url for domain in CREDIBLE_DOMAINS)

def calculate_semantic_similarity(text: str, snippet: str) -> float:
    # Sentence-level chunking for better matching
    text_sents = [sent.text for sent in nlp(text).sents]
    snippet_sents = [sent for sent in re.split(r'[.!?]', snippet) if sent.strip()]
    max_sim = 0.0
    for t in text_sents:
        for s in snippet_sents:
            embeddings = embedder.encode([t, s], convert_to_tensor=True)
            sim = util.cos_sim(embeddings[0], embeddings[1]).item()
            if sim > max_sim:
                max_sim = sim
    return max_sim

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
        # Prefer recent articles (if date available)
        date = result.get("date", "")
        article_entry = {
            "url": url,
            "snippet": snippet,
            "title": title,
            "similarity": round(similarity, 3),
            "date": date
        }
        if credible and similarity >= SIMILARITY_THRESHOLD:
            evidence.append(article_entry)
            credible_and_similar += 1
        else:
            related_articles.append(article_entry)
    score = {0: 0, 1: 50, 2: 70, 3: 85}.get(credible_and_similar, 100)
    is_real = score >= 70
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
