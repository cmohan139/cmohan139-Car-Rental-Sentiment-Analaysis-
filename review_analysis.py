import pandas as pd
from textblob import TextBlob
from keybert import KeyBERT
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')  # Required for TextBlob to tokenize sentences


# Load review data
df = pd.read_csv("data/reviews.csv")

# Initialize KeyBERT
kw_model = KeyBERT()

# Sentiment analysis using TextBlob
def get_sentiment(polarity):
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["Polarity"] = df["review"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["Sentiment"] = df["Polarity"].apply(get_sentiment)

# Keyword extraction using KeyBERT
df["Key Issues"] = df["review"].apply(
    lambda x: ", ".join([kw[0] for kw in kw_model.extract_keywords(x, top_n=2)])
)

# Save summary
df.to_csv("data/review_summary.csv", index=False)
print("✅ Analysis saved to data/review_summary.csv")

# Visualization - Sentiment Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sentiment", palette="pastel")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig("data/sentiment_chart.png")
plt.show()

# Visualization - Top Keywords
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Key Issues'])
word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
word_freq[:10].plot(kind='bar', color='skyblue')
plt.title("Top Mentioned Issues")
plt.ylabel("Frequency")
plt.xlabel("Keywords")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/keyword_chart.png")
plt.show()


