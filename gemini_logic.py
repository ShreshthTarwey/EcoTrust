import os
from google import genai
from google.genai import types
import re

# Load Gemini API Key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize Gemini client
client = genai.Client()

def make_links_clickable(text):
    """
    Converts plain URLs in text to clickable anchor tags.
    """
    # Replace plain URLs with clickable links
    return re.sub(
        r'(https?://[^\s<]+)',
        r'<a href="\1" target="_blank" rel="noopener noreferrer">[source]</a>',
        text
    )

def analyze_with_gemini(news_text):
    try:
        # Prompt to ensure consistent structure in Gemini response
        prompt = (
            "You are a fact-checking assistant. Analyze the following news and respond in this exact format:\n"
            "Confidence Score: <confidence as a percentage, e.g. 87%>\n"
            "Verdict: <Real, Fake, or Uncertain>\n"
            "Explanation: <1-2 sentences explaining your reasoning>\n"
            "Links: <1 or 2 relevant source URLs>\n"
            "News: {}".format(news_text)
        )

        # Add grounding via Google Search
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )

        # Clean output: add <br> for newlines and make URLs clickable
        raw_output = response.text.strip()
        formatted_output = raw_output.replace("\n", "<br>")
        clickable_output = make_links_clickable(formatted_output)

        return clickable_output

    except Exception as e:
        return f"Error during Gemini analysis: {e}"
