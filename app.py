from dotenv import load_dotenv
import os
load_dotenv()
from flask import Flask, render_template, request
from search_logic import score_news
from gemini_logic import analyze_with_gemini

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for fake news detector form
@app.route('/fake-news')
def fake_news_form():
    return render_template('fake_news_detector.html')

# Route to analyze news text
@app.route('/analyze', methods=['POST'])
def analyze():
    news_text = request.form.get('newsText')
    if not news_text:
        return "No newsText submitted.", 400

    # Call keyword + semantic scoring
    result = score_news(news_text)

    # Gemini only if confidence is low
    gemini_summary = None
    if result["confidence"] < 70:
        gemini_summary = analyze_with_gemini(news_text)

    return render_template('result.html',
        news_text=news_text,
        is_real=result["is_real"],
        confidence=result["confidence"],
        evidence=result.get("evidence", []),
        related_articles=result.get("related_articles", []),
        gemini_summary=gemini_summary
    )



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
