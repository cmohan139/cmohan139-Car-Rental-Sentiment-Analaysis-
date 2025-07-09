import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Car Rental Review Analyzer", layout="centered")

st.title("🚗 Car Rental Review Analyzer")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Analyze if not already done
    if "Sentiment" not in df.columns or "Key Issues" not in df.columns:
        st.info("🔍 Analyzing reviews...")

        kw_model = KeyBERT()

        def get_sentiment(polarity):
            if polarity > 0.1:
                return "Positive"
            elif polarity < -0.1:
                return "Negative"
            else:
                return "Neutral"

        df["Polarity"] = df["review"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df["Sentiment"] = df["Polarity"].apply(get_sentiment)
        df["Key Issues"] = df["review"].apply(
            lambda x: ", ".join([kw[0] for kw in kw_model.extract_keywords(str(x), top_n=2)])
        )

    # Display data
    st.subheader("📋 Review Summary")
    st.dataframe(df[["review", "Sentiment", "Key Issues"]])

    # Download link
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Result CSV", data=csv, file_name="review_summary.csv", mime="text/csv")

    # Charts
    st.subheader("📊 Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Sentiment", palette="pastel", ax=ax1)
    st.pyplot(fig1)

    st.subheader("🔑 Top Mentioned Key Issues")
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["Key Issues"])
    word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    word_freq[:10].plot(kind="bar", color="skyblue", ax=ax2)
    ax2.set_title("Top Key Issues")
    st.pyplot(fig2)
