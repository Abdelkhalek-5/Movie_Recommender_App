import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# === PAGE CONFIG ===
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# === HEADER ===
st.markdown("## 🎥 Content-Based Movie Recommender")
st.markdown("Get movie recommendations based on genre similarity.")

@st.cache_data
def load_data():
    try:
        df = pd.read_json("Data/movie_data.json")
        # Ensure 'genres' is a list before joining, else convert to an empty string
        df['genres'] = df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x) if x is not None else '')
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return pd.DataFrame()

# Load data and check if it is empty
df = load_data()

if df.empty:
    st.stop()

# === TF-IDF + COSINE SIMILARITY ===
@st.cache_data
def build_model(df, feature_col='genres'):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[feature_col])
    cosine_sim = cosine_similarity(tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_model(df)

# === RECOMMENDER FUNCTION ===
def recommend(title, k=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genres']]

# === STREAMLIT UI ===
st.markdown("### 🎞️ Select a movie you like:")
movie_name = st.selectbox("Choose a movie:", sorted(df['title'].unique()))

if st.button("Recommend 🎯"):
    recs = recommend(movie_name)
    if recs is not None and not recs.empty:
        st.success("Here are 5 movies you might enjoy:")
        for i, row in recs.iterrows():
            st.markdown(f"**🎬 {row['title']}**  \n🧬 Genres: _{row['genres']}_")
    else:
        st.warning("No recommendations found.")
