import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pickle
import base64
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Ensure NLTK stopwords are downloaded
import nltk
nltk.download('stopwords')

# Constants and Setup
STOPWORDS = set(stopwords.words("english"))

# Function to load models and tools
def load_models(model_name):
    # Load pickled models and tools based on model_name
    if model_name == "Decision Tree":
        with open("Models/model_dt.pkl", "rb") as f:
            predictor = pickle.load(f)
    elif model_name == "XGBoost":
        with open("Models/model_xgb.pkl", "rb") as f:
            predictor = pickle.load(f)
    else:
        raise ValueError("Model not supported.")

    with open("Models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("Models/countVectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    return predictor, scaler, cv

# Function for single prediction
def single_prediction(predictor, scaler, cv, text_input):
    stemmer = PorterStemmer()
    corpus = []
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    return "Positive" if y_predictions == 1 else "Negative"

# Function for bulk prediction and graph generation
def bulk_prediction_and_graph(predictor, scaler, cv, data):
    stemmer = PorterStemmer()
    corpus = []
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    data["Predicted sentiment"] = ["Positive" if pred == 1 else "Negative" for pred in y_predictions]

    # Generate graph
    graph = get_distribution_graph(data)
    return data, graph

# Function to generate sentiment distribution graph
def get_distribution_graph(data):
    # Create figure and axes for subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate sentiment counts
    sentiment_counts = data["Predicted sentiment"].value_counts()
    labels = sentiment_counts.index.tolist()
    sizes = sentiment_counts.values.tolist()

    # Colors and explode settings for pie chart
    colors = ["lightgreen", "lightcoral"]
    explode = (0.1, 0)

    # Plot 1: Pie chart for sentiment distribution
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title("Sentiment Distribution (Pie Chart)", pad=20, fontsize=16, fontweight='bold')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Plot 2: Bar plot for sentiment counts
    ax2.bar(labels, sizes, color=colors)
    ax2.set_title("Sentiment Distribution (Bar Plot)", pad=20, fontsize=16, fontweight='bold')
    ax2.set_xlabel("Sentiment")
    ax2.set_ylabel("Count")

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Convert plot to PNG format and save to BytesIO object
    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

# Main Streamlit app
def main():
    st.set_page_config(page_title="Amazon Alexa Sentiment Analysis", layout="wide")
    st.title("Amazon Alexa Review Sentiment Analysis")

    # Sidebar - Landing page or any introductory content
    st.sidebar.title("Welcome")
    st.sidebar.write("This is a Streamlit app for sentiment analysis.")
    st.sidebar.write("Upload a CSV file or enter text for prediction.")

    # Select model
    model_name = st.sidebar.selectbox("Select Model", ["Decision Tree", "XGBoost"])

    # Load models and tools based on selected model
    predictor, scaler, cv = load_models(model_name)

    # Page selection
    pages = ["Home", "Bulk Prediction", "Single Prediction"]
    page = st.sidebar.selectbox("Select a page", pages)

    # Home page
    if page == "Home":
        st.header("Home Page")
        st.write("This is the landing page of your sentiment analysis app.")
        

    # Bulk Prediction page
    elif page == "Bulk Prediction":
        st.header("Bulk Prediction")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if "feedback" in data.columns:
                data.rename(columns={"feedback": "Sentence"}, inplace=True)  # Assuming "feedback" should be "Sentence"
            if "Sentence" in data.columns:
                predictions, graph = bulk_prediction_and_graph(predictor, scaler, cv, data)
                st.subheader("Predictions")
                st.write(predictions)
                st.subheader("Sentiment Distribution")
                st.image(graph, use_column_width=True)
            else:
                st.warning("CSV file must contain a column named 'Sentence'.")

    # Single Prediction page
    elif page == "Single Prediction":
        st.header("Single Prediction")
        text_input = st.text_area("Enter a text for sentiment analysis")
        if st.button("Predict"):
            if text_input.strip() != "":
                predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
                st.success(f"Predicted Sentiment: {predicted_sentiment}")
            else:
                st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
