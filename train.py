import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
df = pd.read_csv('imdb_dataset.csv')

# Prepare data
X = df['review']
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0) # Convert sentiment to numerical (1 for positive, 0 for negative)

# Split data (using a small subset for faster training as per instructions)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use a smaller subset for training (e.g., 5000 samples)
subset_size = 5000
X_train_subset = X_train.sample(n=min(subset_size, len(X_train)), random_state=42)
y_train_subset = y_train.loc[X_train_subset.index]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features for efficiency
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_subset)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000) # Increase max_iter for convergence
model.fit(X_train_tfidf, y_train_subset)

# Save the trained model and vectorizer
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print('Model and TF-IDF vectorizer trained and saved successfully.')



