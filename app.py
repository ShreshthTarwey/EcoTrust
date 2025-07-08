from flask import Flask, render_template, request
from search_logic import score_news
from gemini_logic import analyze_with_gemini
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fake-news", methods=["GET", "POST"])
def fake_news_detector():
    if request.method == "POST":
        news_text = request.form["news_text"]

        result = score_news(news_text)
        is_real = result["is_real"]
        confidence = result["confidence"]
        matched_sources = result["evidence"]
        verdict = "Likely Real News" if is_real else "Likely Fake News"

        gemini_explanation = analyze_with_gemini(news_text)
        if gemini_explanation:
            gemini_explanation = gemini_explanation.replace("\n", "<br>")

        return render_template(
            "result.html",
            news_text=news_text,
            verdict=verdict,
            confidence=confidence,
            is_real=is_real,
            matched_sources=matched_sources,
            evidence=matched_sources,
            related_articles=result.get("related_articles", []),
            gemini_summary=gemini_explanation
        )

    return render_template("fake_news_detector.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    news_text = request.form["news_text"]

    result = score_news(news_text)
    is_real = result["is_real"]
    confidence = result["confidence"]
    matched_sources = result["evidence"]
    verdict = "Likely Real News" if is_real else "Likely Fake News"

    gemini_explanation = analyze_with_gemini(news_text)
    if gemini_explanation:
        gemini_explanation = gemini_explanation.replace("\n", "<br>")

    return render_template(
        "result.html",
        news_text=news_text,
        verdict=verdict,
        confidence=confidence,
        is_real=is_real,
        matched_sources=matched_sources,
        evidence=matched_sources,
        related_articles=result.get("related_articles", []),
        gemini_summary=gemini_explanation
    )

# For Vercel runtime compatibility
def handler(environ, start_response):
    return app.wsgi_app(environ, start_response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
