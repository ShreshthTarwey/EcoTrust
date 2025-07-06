# gemini_logic.py

import os
from google import genai

# Ensure API key is loaded
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize client using env variable (automatically picked up)
client = genai.Client()

def analyze_with_gemini(news_text):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Is the following news real or fake? Provide a short explanation.\n\nNews: {news_text}"
        )
        return response.text.strip()
    except Exception as e:
        return f"Error during Gemini analysis: {e}"
