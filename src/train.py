import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from src.clean import clean_text

# Load and preprocess data
df = pd.read_csv('data/spam.csv', sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# Model training
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'models/spam_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("Training complete. Model saved to 'models/spam_model.pkl'")