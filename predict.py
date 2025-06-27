import joblib
import sys
import pandas as pd

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<review_text>\"")
        sys.exit(1)

    review = sys.argv[1]
    sentiment, confidence = predict_sentiment(review)

    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")



