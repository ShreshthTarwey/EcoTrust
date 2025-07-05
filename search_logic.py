import re
import random
import requests
from typing import List
import spacy
nlp = spacy.load("en_core_web_sm")
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

try:
    from rake_nltk import Rake
except ImportError:
    Rake = None  # Fallback if rake_nltk is not installed

# List of credible domains to consider
CREDIBLE_DOMAINS = [
    "bbc.com",
    "reuters.com",
    "ndtv.com",
    "indianexpress.com",
    "thehindu.com"
]
def extract_named_entities(text: str, labels=("ORG", "PERSON", "GPE")) -> List[str]:
    """
    Extracts named entities like organizations, people, locations using SpaCy.
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in labels]
    return list(set(entities))

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    keywords = []

    # Use RAKE
    if Rake:
        rake = Rake()
        rake.extract_keywords_from_text(text)
        raw_keywords = rake.get_ranked_phrases()
        cleaned_keywords = [kw for kw in raw_keywords if len(kw.split()) > 1 or len(kw) > 5]
        keywords.extend(cleaned_keywords)

    # Use SpaCy NER
    named_entities = extract_named_entities(text)
    keywords.extend(named_entities)

    # Remove duplicates, limit to top `num_keywords`
    final_keywords = list(set(keywords))[:num_keywords]
    return final_keywords


def real_web_search(keywords: List[str], num_results: int = 7) -> List[str]:
    """
    Performs real web search using Google Custom Search API.
    Returns a list of result URLs.
    """
    query = " ".join(keywords)
    url = (
        f"https://www.googleapis.com/customsearch/v1?"
        f"key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&num={num_results}"
    )
    try:
        response = requests.get(url)
        results = response.json()
        urls = [item["link"] for item in results.get("items", [])]
        return urls
    except Exception as e:
        print("Search error:", e)
        return []


def count_credible_links(urls: List[str]) -> int:
    """
    Counts how many URLs are from credible domains.
    """
    count = 0
    for url in urls:
        for domain in CREDIBLE_DOMAINS:
            if domain in url:
                count += 1
                break
    return count

def keyword_match_score(text: str) -> dict:
    """
    Extracts keywords, performs real web search, and returns a confidence score (0-100)
    along with is_real (True/False).
    """
    keywords = extract_keywords(text)
    print("Extracted keywords:", keywords)
    search_results = real_web_search(keywords, num_results=7)
    print("Search Results:", search_results)
    credible_count = count_credible_links(search_results)

    # Score mapping
    if credible_count == 0:
        score = 0
    elif credible_count == 1:
        score = 30
    elif credible_count == 2:
        score = 50
    elif credible_count == 3:
        score = 70
    elif credible_count == 4:
        score = 85
    else:
        score = 100

    is_real = score >= 70  # threshold for calling it "likely real"

    return {
        "is_real": is_real,
        "confidence": score
    }



# Example usage (remove or comment out in production)
if __name__ == "__main__":
    text = "India’s economy is growing steadily according to BBC and The Hindu."
    print("Result:", keyword_match_score(text))
