import joblib
import pandas as pd
from preprocess import clean_text

class SentimentModel:
    def __init__(self):
        # Load the trained model and vectorizer
        self.model = joblib.load('models/sentiment_model.pkl')
        self.vectorizer = pd.read_pickle('data/vectorizer.pkl')
    
    def predict(self, text):
        # Clean text
        cleaned_text = clean_text(text)
        # Transform text to TF-IDF features
        text_tfidf = self.vectorizer.transform([cleaned_text])
        # Predict sentiment
        prediction = self.model.predict(text_tfidf)[0]
        # Get probability scores
        proba = self.model.predict_proba(text_tfidf)[0]
        
        # Return result
        if prediction == 0:
            category = "Baseball"
        else:
            category = "Medical"
            
        return {
            'category': category,
            'confidence': float(proba[prediction]),
            'baseball_probability': float(proba[0]),
            'medical_probability': float(proba[1])
        }