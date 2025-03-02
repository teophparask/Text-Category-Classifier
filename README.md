# Text Category Classifier

A machine learning application that classifies text as either baseball-related or medical-related content.

## Project Overview

This project uses Natural Language Processing (NLP) and machine learning to analyze text and determine if it's related to baseball or medicine. The application:

1. Preprocesses text data using NLTK
2. Uses TF-IDF vectorization to convert text to features
3. Trains a Logistic Regression model
4. Provides a web interface for real-time text classification

## Technologies Used

- Python 3.8+
- scikit-learn for machine learning
- NLTK for text processing
- Flask for the web application
- Pandas & NumPy for data handling
- Matplotlib for visualizations

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open your browser and go to http://127.0.0.1:5000/

## Model Performance

The model achieves approximately 90% accuracy on the test dataset. The confusion matrix and detailed performance metrics are available in the application.

## Future Improvements

- Add more categories
- Implement more sophisticated NLP techniques
- Use a more advanced model (e.g., BERT)
- Improve the UI/UX