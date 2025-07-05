from flask import Flask, render_template, request
from search_logic import keyword_match_score

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for fake news detector form
@app.route('/fake-news')
def fake_news_form():
    return render_template('fake_news_detector.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    news_text = request.form.get('newsText')
    if not news_text:
        return "No newsText submitted.", 400

    # Fake detection logic
    is_real = "fake" not in news_text.lower()
    confidence = 90 if not is_real else 85

    result = keyword_match_score(news_text)
    data = {
    "newsText": news_text,
    "isReal": result["is_real"],
    "confidence": result["confidence"]
}
    return render_template('result.html', data=data, 
                       newsText=news_text,
                       isReal=result["is_real"],
                       confidence=result["confidence"])


# The /result route is removed because result data is passed directly from /analyze


# (Optional) Run the app directly
if __name__ == '__main__':
    app.run(debug=True)
