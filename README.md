# 🤖 Amazon Product Review Sentiment Analyzer

A simple web app using **BERT** to predict sentiment from Amazon product reviews.

---

## 🔍 Features

- Takes a product review as input
- Predicts if sentiment is Positive, Neutral, or Negative
- Uses pretrained BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`)
- Built with **Streamlit**

---

## 🚀 How to Run

### 1. Clone this repository:
```bash
git clone https://github.com/YOUR-USERNAME/Sentiment-Analysis.git
cd Sentiment-Analysis
2. Create and activate virtual environment:

python -m venv sentiment-env
sentiment-env\Scripts\activate  # On Windows
3. Install dependencies:

pip install -r requirements.txt
4. Run the Streamlit app:

streamlit run app.py