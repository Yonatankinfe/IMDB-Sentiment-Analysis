# IMDB-Sentiment-Analysis
This project provides a basic machine learning pipeline to classify movie reviews as positive or negative using scikit-learn.
IMDB Sentiment Analysis

This project provides a basic machine learning pipeline to classify movie reviews as positive or negative using scikit-learn.

## Installation

To set up the project, first clone the repository and then install the required dependencies:
```bash
pip install -r requirements.txt
```
## Training the Model

The model is a Logistic Regression classifier trained on TF-IDF vectorized movie reviews. To train the model, run the train.py script:

```bash
python train.py
```
This will train the model and save the trained model (logistic_regression_model.pkl) and the TF-IDF vectorizer (tfidf_vectorizer.pkl) to disk.

## Running Predictions

To predict the sentiment of a new movie review, use the predict.py script. Provide the review text as a command-line argument:

```Bash
python predict.py "I loved this movie! It was fantastic and I highly recommend it."
```
The script will output whether the sentiment is "positive" or "negative" and a confidence score.

+ Example:
```bash
Sentiment: positive
Confidence: 0.9605
```
# Flask API

For convenience, a Flask API wrapper is also provided. To start the API server:
```bash
python app.py
```
The API will be available at http://localhost:5000.

## API Endpoints

+ GET /: Returns API information

+ POST /predict: Predicts sentiment for a given review
  
Example API usage:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was terrible! I hated it."}'
```

```bash
{
  "confidence": 0.8636,
  "review": "This movie was terrible! I hated it.",
  "sentiment": "negative"
}
```
### Model Details

* Algorithm: Logistic Regression


* Vectorization: TF-IDF with max 5000 features

* Training Data: 5000 samples from the IMDB movie review dataset

*Classes: Binary classification (positive/negative sentiment)


