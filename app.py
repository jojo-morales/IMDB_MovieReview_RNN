import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('tuned_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Constants
# Maximum length of sequences
MAX_LEN = 200  

def preprocess(text):
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded_sequence

def predict_sentiment(review_text):
    processed_text = preprocess(review_text)
    #evaluate = model.evaluate(processed_text)
    prediction = model.predict(processed_text) 
    print(f"Prediction: {prediction}")  
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    return sentiment

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="", layout="centered")

st.title("IMDB Review Sentiment Analysis")

st.markdown("""
    IMDB Review Sentiment Analysis app! 
    - This uses a Recurrent Neural Network (RNN) to classify movie reviews as either positive or negative. 
    - Enter a review in the text box below and click on the "Predict Sentiment" button to see the result.
""")

review_text = st.text_area("Enter Movie Review:", placeholder="Type your movie review here...", height=150)

if st.button('Predict Sentiment'):
    if review_text.strip() == "":
        st.error("Please enter a review before predicting.")
    else:
        sentiment = predict_sentiment(review_text)
        st.subheader(f"Sentiment: **{sentiment}**")
