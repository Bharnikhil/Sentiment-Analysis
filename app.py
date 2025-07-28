# app.py

import streamlit as st
from bert_sentiment import get_sentiment

st.set_page_config(page_title="BERT Sentiment Analyzer", page_icon="🤖")

st.title("🤖 Amazon Product Review Sentiment Analyzer (BERT Powered)")
st.write("Type your review below and get a sentiment prediction powered by BERT:")

review_text = st.text_area("📝 Write your review here:")

if st.button("🔍 Analyze"):
    if review_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, confidence = get_sentiment(review_text)
        st.success(f"**Sentiment:** `{sentiment}` with `{confidence*100:.1f}%` confidence")
