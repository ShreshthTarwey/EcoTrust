import os
import re
import google.generativeai as genai

# Load Gemini API Key and configure
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

def make_links_clickable(text: str) -> str:
    """
    Converts URLs in plain text to HTML clickable anchor tags.
    """
    return re.sub(
        r'(https?://[^\s<]+)',
        r'<a href="\1" target="_blank" rel="noopener noreferrer">[source]</a>',
        text
    )

def analyze_with_gemini(news_text: str) -> str:
    """
    Sends the news text to Gemini for analysis and formats the response with links and HTML breaks.
    """
    try:
        prompt = (
            "You are a fact-checking assistant. Analyze the following news and respond in this exact format:\n"
            "Confidence Score: <confidence as a percentage, e.g. 87%>\n"
            "Verdict: <Real, Fake, or Uncertain>\n"
            "Explanation: <1-2 sentences explaining your reasoning>\n"
            "Links: <1 or 2 relevant source URLs>\n"
            f"News: {news_text}"
        )

        response = model.generate_content(prompt)
        raw_output = response.text.strip()

        clickable = make_links_clickable(raw_output)
        html_ready_output = clickable.replace("\n", "<br>")

        return html_ready_output

    except Exception as e:
        return f"❌ Error during Gemini analysis: {str(e)}"
