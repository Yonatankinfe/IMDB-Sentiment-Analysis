from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and vectorizer
model = joblib.load("logistic_regression_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_sentiment(review_text):
    # Vectorize the input review
    review_tfidf = tfidf_vectorizer.transform([review_text])

    # Predict sentiment
    prediction = model.predict(review_tfidf)[0]
    prediction_proba = model.predict_proba(review_tfidf)[0]

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = prediction_proba[prediction]

    return sentiment, confidence

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "IMDB Sentiment Analysis API",
        "usage": "POST /predict with JSON body: {'review': 'your review text'}"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'review' not in data:
            return jsonify({"error": "Please provide 'review' in the request body"}), 400
        
        review_text = data['review']
        sentiment, confidence = predict_sentiment(review_text)
        
        return jsonify({
            "review": review_text,
            "sentiment": sentiment,
            "confidence": float(confidence)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


