from flask import Flask, render_template, request
import os
import sys
from model import SentimentModel

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    text = ""
    
    if request.method == 'POST':
        text = request.form['text']
        model = SentimentModel()
        result = model.predict(text)
    
    return render_template('index.html', result=result, text=text)

if __name__ == '__main__':
    # Check if model exists, if not run training
    if not os.path.exists('models/sentiment_model.pkl'):
        print("Model not found. Running preprocessing and training...")
        from preprocess import preprocess_data
        from train import train_model
        
        preprocess_data()
        train_model()
    
    app.run(debug=True)