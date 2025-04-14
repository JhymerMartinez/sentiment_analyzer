import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Example data for training
texts = [
  "I love this product",
  "it's horrible",
  "it's okay",
  "very bad",
  "I recommend it",
  "I didn't like it",
  "terrible service",
  "excellent quality",
  "so-so",
  "very good support"
]
labels = [
  "positive",
  "negative",
  "neutral",
  "negative",
  "positive",
  "negative",
  "negative",
  "positive",
  "neutral",
  "positive"
]

# Train model
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)

def generate_recommendation(sentiment: str) -> str:
  if sentiment == "negative":
    return "Review common aspects like delivery, quality, or service."
  elif sentiment == "neutral":
    return "Consider improving the experience to stand out more."
  elif sentiment == "positive":
    return "Keep reinforcing what you're doing well."
  return ""

# Streamlit interface
st.title("Sentiment Analyzer with Recommendations")
comment = st.text_area("Enter a comment to analyze")
if st.button("Analyze"):
  if comment.strip():
    sentiment = model.predict([comment])[0]
    recommendation = generate_recommendation(sentiment)
    st.success(f"**Sentiment:** {sentiment}")
    st.info(f"**Recommendation:** {recommendation}")
  else:
    st.warning("Please enter a comment.")
