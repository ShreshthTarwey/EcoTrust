from flask import Flask, render_template, request
from search_logic import score_news
from gemini_logic import analyze_with_gemini

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fake-news", methods=["GET", "POST"])
def fake_news_detector():
    if request.method == "POST":
        news_text = request.form["news_text"]

        # Run the credibility scoring logic
        result = score_news(news_text)
        is_real = result["is_real"]
        confidence = result["confidence"]
        matched_sources = result["evidence"]
        verdict = "Likely Real News" if is_real else "Likely Fake News"

        # Run Gemini explanation (called only once)
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

    # Run the credibility scoring logic
    result = score_news(news_text)
    is_real = result["is_real"]
    confidence = result["confidence"]
    matched_sources = result["evidence"]
    verdict = "Likely Real News" if is_real else "Likely Fake News"

    # Run Gemini explanation (called only once)
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

if __name__ == "__main__":
    app.run(debug=True)
