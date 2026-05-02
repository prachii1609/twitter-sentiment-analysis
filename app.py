import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

# Title
st.title("💬 Twitter Sentiment Analysis")
st.write("Enter a tweet and check whether it is Positive, Negative or Neutral")

# Input box
text = st.text_area("✍️ Enter your tweet here:")

# Prediction function
def predict_sentiment(text):
    text = text.lower()
    vector = cv.transform([text])
    return model.predict(vector)[0]

# Button
if st.button("🔍 Analyze Sentiment"):
    if text.strip() != "":
        result = predict_sentiment(text)

        # Display result with colors
        if result == "Positive":
            st.success(f"😊 Sentiment: {result}")
        elif result == "Negative":
            st.error(f"😡 Sentiment: {result}")
        else:
            st.info(f"😐 Sentiment: {result}")

    else:
        st.warning("⚠️ Please enter some text")
