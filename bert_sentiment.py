# bert_sentiment.py

from transformers import pipeline

# Load the pre-trained BERT model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

def get_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']  # e.g., "4 stars"
    score = result['score']  # Confidence

    # Map star rating to general sentiment
    if "1" in label or "2" in label:
        sentiment = "Negative"
    elif "3" in label:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return sentiment, round(score, 3)
