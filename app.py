import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Function to perform sentiment analysis using transformer model
def analyze_sentiment_transformer(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    sentiment = result['label']
    score = result['score']

    return sentiment, score

# Function for EDA
def perform_eda(data):
    st.subheader("Exploratory Data Analysis (EDA)")

    # Show first few rows of the dataset
    st.write("First few rows of the dataset:")
    st.write(data.head())

    # Show basic statistics
    st.write("Basic Statistics:")
    st.write(data.describe())

    # Plot distribution of sentiments
    fig = px.histogram(data, x='Sentiment', title='Sentiment Distribution')
    st.plotly_chart(fig)

# Streamlit app
def main():
    st.title("Sentiment Analysis and EDA App with Transformer Model")

    # Sidebar
    st.sidebar.header("Choose an Option")
    option = st.sidebar.radio("", ["Single Text Sentiment Analysis", "CSV Dataset EDA"])

    # Main content
    if option == "Single Text Sentiment Analysis":
        st.header("Single Text Sentiment Analysis")
        text_input = st.text_area("Enter a text:")
        
        if st.button("Analyze"):
            sentiment, score = analyze_sentiment_transformer(text_input)
            st.write("Sentiment:", sentiment)
            st.write("Confidence Score:", score)

    elif option == "CSV Dataset EDA":
        st.header("CSV Dataset EDA")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            data['Sentiment'], data['Confidence'] = zip(*data['Text'].apply(analyze_sentiment_transformer))

            perform_eda(data)

# Run the app
if __name__ == "__main__":
    main()
