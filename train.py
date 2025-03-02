import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_model():
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train_tfidf, y_train = pd.read_pickle('data/train_data.pkl')
    X_test_tfidf, y_test = pd.read_pickle('data/test_data.pkl')
    
    # Train model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Baseball', 'Medical']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Baseball', 'Medical'], rotation=45)
    plt.yticks(tick_marks, ['Baseball', 'Medical'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save figure
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/confusion_matrix.png')
    
    # Save model
    print("Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/sentiment_model.pkl')
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    train_model()