import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def download_dataset():
    """Download IMDB dataset if not already downloaded"""
    # For this example, we'll use a smaller version from scikit-learn
    from sklearn.datasets import load_files
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("Using scikit-learn's built-in movie review dataset")
    # Return the dataset path
    return 'data/imdb_reviews'

def clean_text(text):
    """Clean and preprocess text"""
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove non-letters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Split into words
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    # Join words back into text
    text = ' '.join(words)
    return text

def preprocess_data():
    """Load, clean and split the dataset"""
    # Load data
    from sklearn.datasets import fetch_20newsgroups
    categories = ['rec.sport.baseball', 'sci.med']
    
    print("Loading dataset...")
    train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    
    # Clean text
    print("Cleaning text...")
    X_train = [clean_text(text) for text in train_data.data]
    X_test = [clean_text(text) for text in test_data.data]
    y_train = train_data.target
    y_test = test_data.target
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save processed data
    print("Saving processed data...")
    os.makedirs('data', exist_ok=True)
    pd.to_pickle((X_train_tfidf, y_train), 'data/train_data.pkl')
    pd.to_pickle((X_test_tfidf, y_test), 'data/test_data.pkl')
    pd.to_pickle(vectorizer, 'data/vectorizer.pkl')
    
    print("Preprocessing complete!")
    return X_train_tfidf, y_train, X_test_tfidf, y_test, vectorizer

if __name__ == "__main__":
    download_dataset()
    preprocess_data()