# sentiment_model.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_prepare_data():
    df = pd.read_csv("Reviews.csv")
    df = df[['Text', 'Score']].dropna()
    
    # Map scores to sentiments
    df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))

    # Remove neutral reviews
    df = df[df['Sentiment'] != 'neutral']

    # Clean the text
    df['Clean_Text'] = df['Text'].apply(preprocess_text)
    return df

def train_model():
    df = load_and_prepare_data()
    X = df['Clean_Text']
    y = df['Sentiment']

    # Build pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', MultinomialNB())
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save model
    joblib.dump(model, "sentiment_model.pkl")
    print("✅ Model saved as `sentiment_model.pkl`")

# Run training
if __name__ == "__main__":
    train_model()
