import os
import re
import requests
from typing import List, Dict
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from scipy.spatial.distance import cosine

# Load models once
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and accurate
kw_model = KeyBERT()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Config
CREDIBLE_DOMAINS = [
    "bbc.com", "reuters.com", "ndtv.com", "indianexpress.com", "thehindu.com",
    "timesofindia.indiatimes.com", "hindustantimes.com", "thewire.in", "scroll.in",
    "economictimes.indiatimes.com", "moneycontrol.com", "livemint.com", "news18.com",
    "business-standard.com", "cnn.com", "nytimes.com", "aljazeera.com", "theguardian.com",
    "abcnews.go.com", "npr.org", "cbsnews.com"
]
SIMILARITY_THRESHOLD = 0.60
MAX_RELATED_ARTICLES = 5
MAX_SENTENCES = 2

def extract_keywords(text: str) -> List[str]:
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5
    )
    if keywords:
        return [kw[0] for kw in keywords]
    
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "GPE", "EVENT"}]
    return ents or text.split()[:5]

def real_web_search(keywords: List[str], num_results: int = 10, days: int = 7) -> List[Dict]:
    query = " ".join(keywords)
    serpapi_url = f"https://serpapi.com/search.json?q={query}&engine=google&api_key={SERPAPI_KEY}&num={num_results}&tbs=qdr:d{days}"
    
    try:
        response = requests.get(serpapi_url, timeout=5)
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
        print("⚠️ SerpAPI Error:", e)

    # Fallback to Google Custom Search
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&num={num_results}"
    try:
        response = requests.get(url, timeout=5)
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
        print("⚠️ Google Search Error:", e)
        return []

def is_credible_domain(url: str) -> bool:
    return any(domain in url for domain in CREDIBLE_DOMAINS)

def calculate_semantic_similarity(text: str, snippet: str) -> float:
    try:
        text_sents = [sent.text.strip() for sent in nlp(text).sents][:MAX_SENTENCES]
        snippet_sents = [s.strip() for s in re.split(r'[.!?]', snippet) if s.strip()][:MAX_SENTENCES]
        all_sentences = text_sents + snippet_sents
        embeddings = embedder.encode(all_sentences, convert_to_numpy=True)

        max_sim = 0.0
        for i in range(len(text_sents)):
            for j in range(len(text_sents), len(all_sentences)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                max_sim = max(max_sim, sim)
        return max_sim
    except Exception as e:
        print("⚠️ Similarity Error:", e)
        return 0.0

def score_news(text: str) -> dict:
    keywords = extract_keywords(text)
    print("🔍 Extracted Keywords:", keywords)

    results = real_web_search(keywords)
    print("📄 Search Results:", results)

    evidence = []
    related_articles = []
    credible_and_similar = 0

    for result in results:
        url = result["url"]
        snippet = result["snippet"]
        title = result["title"]
        similarity = calculate_semantic_similarity(text.lower(), snippet)
        date = result.get("date", "")

        article = {
            "url": url,
            "snippet": snippet,
            "title": title,
            "similarity": round(similarity, 3),
            "date": date
        }

        if is_credible_domain(url) and similarity >= SIMILARITY_THRESHOLD:
            evidence.append(article)
            credible_and_similar += 1
        else:
            related_articles.append(article)

    score = {0: 0, 1: 50, 2: 70, 3: 85}.get(credible_and_similar, 100)
    is_real = score >= 70

    related_articles = sorted(related_articles, key=lambda x: -x["similarity"])[:MAX_RELATED_ARTICLES]

    return {
        "is_real": is_real,
        "confidence": score,
        "evidence": evidence,
        "related_articles": related_articles
    }

# Local test
if __name__ == "__main__":
    sample_news = "Ameesha Patel is planning to marry Salman Khan and have good-looking babies together."
    print("📰 Input:", sample_news)
    print("🧠 Output:", score_news(sample_news))
