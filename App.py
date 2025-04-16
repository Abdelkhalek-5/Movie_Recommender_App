import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load your processed movie data
movies = pd.read_json("Data/movie_data.json")

# Combine features into a single string for each movie
movies['combined'] = movies['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Vectorize the text
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Map movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Content-based recommend function
def recommend(title, num_recommendations=5):
    if title not in indices:
        return []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommender (Content-Based)")
movie_name = st.selectbox("Choose a movie:", sorted(movies['title'].unique()))

if st.button("Get Recommendations"):
    results = recommend(movie_name)
    if results.empty:
        st.warning("No similar movies found.")
    else:
        st.subheader("You may also like:")
        for _, row in results.iterrows():
            st.markdown(f"- **{row['title']}** — Genres: _{', '.join(row['genres'])}_")
