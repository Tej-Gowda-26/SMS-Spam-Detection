import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from src.clean import clean_text

model = joblib.load('models/spam_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_sms(text):
    cleaned_text = clean_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    probs = model.predict_proba(text_vec)[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    return label, probs[1], probs[0] 
