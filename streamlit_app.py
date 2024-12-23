import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from transformers import pipeline
import streamlit as st

# Function to extract reviews from Amazon product URL
def extract_reviews(url, max_pages=1):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    reviews = []
    page = 1

    while page <= max_pages:
        response = requests.get(f"{url}&pageNumber={page}", headers=headers)
        if response.status_code != 200:
            st.warning(f"Failed to fetch page {page}. Status code: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        review_divs = soup.find_all("div", {"data-hook": "review"})

        for div in review_divs:
            try:
                review_text = div.find("span", {"data-hook": "review-body"}).text.strip()
                star_rating = div.find("i", {"data-hook": "review-star-rating"}).text.strip()
                reviews.append({"review": review_text, "star_rating": star_rating})
            except AttributeError:
                continue

        page += 1

    return pd.DataFrame(reviews)

# Title
st.title("Product Review Analysis")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Extract Reviews", "Review Summarization", "Sentiment Analysis", "Recommender System"])

# Review Extraction
if option == "Extract Reviews":
    st.header("Extract Reviews ")

    url_input = st.text_input("Enter Product URL:")

    if st.button("Extract Reviews"):
        if url_input:
            st.info("Extracting reviews. This may take a few moments...")
            extracted_reviews = extract_reviews(url_input, max_pages=5)

            if not extracted_reviews.empty:
                st.success("Reviews extracted successfully!")
                st.write(extracted_reviews)

                # Save to CSV
                csv_file = "amazon_reviews.csv"
                extracted_reviews.to_csv(csv_file, index=False)
                st.download_button("Download Reviews as CSV", data=open(csv_file, "rb"), file_name=csv_file)
            else:
                st.error("No reviews found or failed to fetch reviews.")
        else:
            st.warning("Please enter a valid Amazon product URL.")

# Review Summarization
elif option == "Review Summarization":
    st.header("Review Summarization")

    if "data" not in st.session_state:
        st.warning("Please upload or extract a dataset first!")
    else:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

        reviews = st.session_state["data"]["review"].dropna().tolist()
        summary = summarizer(" ".join(reviews[:500]), max_length=130, min_length=30, do_sample=False)
        st.write("Summarized Review:")
        st.write(summary[0]['summary_text'])

# Sentiment Analysis
elif option == "Sentiment Analysis":
    st.header("Sentiment Analysis")

    if "data" not in st.session_state:
        st.warning("Please upload or extract a dataset first!")
    else:
        sentiment_analyzer = pipeline("sentiment-analysis", model="roberta-base")

        st.write("Sentiment Analysis Results:")
        df = st.session_state["data"]
        df["sentiment"] = df["review"].dropna().apply(lambda x: sentiment_analyzer(x)[0]['label'])
        st.write(df[["review", "sentiment"]])

# Recommender System
elif option == "Recommender System":
    st.header("Star-Based Recommender System")

    if "data" not in st.session_state:
        st.warning("Please upload or extract a dataset first!")
    else:
        df = st.session_state["data"]

        # Extract ratings
        if "star_rating" in df.columns:
            avg_rating = np.mean(df["star_rating"].astype(float))
            st.metric("Average Rating", f"{avg_rating:.2f}")

            top_reviews = df.nlargest(5, "star_rating")[["review", "star_rating"]]
            st.write("Top Reviews:")
            st.write(top_reviews)
        else:
            st.warning("Star rating column not found in the dataset!")

st.sidebar.info("Developed for quick review analysis and insights.")


