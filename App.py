import streamlit as st
from PIL import Image
import json
from bs4 import BeautifulSoup
import requests, io
import PIL.Image
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load data
@st.cache_data
def load_data():
    with open('./Data/movie_data.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
    with open('./Data/movie_titles.json', 'r+', encoding='utf-8') as f:
        movie_titles = json.load(f)
    return data, movie_titles

hdr = {'User-Agent': 'Mozilla/5.0'}

# Fetch movie poster
def movie_poster_fetcher(imdb_link):
    url_data = requests.get(imdb_link, headers=hdr).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_dp = s_data.find("meta", property="og:image")
    movie_poster_link = imdb_dp.attrs['content']
    u = urlopen(movie_poster_link)
    raw_data = u.read()
    image = PIL.Image.open(io.BytesIO(raw_data))
    image = image.resize((158, 301), )
    st.image(image, use_column_width=False)

# Get movie information
def get_movie_info(imdb_link):
    url_data = requests.get(imdb_link, headers=hdr).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_content = s_data.find("meta", property="og:description")
    movie_descr = imdb_content.attrs['content']
    movie_descr = str(movie_descr).split('.')
    movie_director = movie_descr[0]
    movie_cast = str(movie_descr[1]).replace('With', 'Cast: ').strip()
    movie_story = 'Story: ' + str(movie_descr[2]).strip() + '.'
    rating = s_data.find("span", class_="sc-bde20123-1 iZlgcd").text
    movie_rating = 'Total Rating count: ' + str(rating)
    return movie_director, movie_cast, movie_story, movie_rating

# Content-based recommendation system
def content_based_recommender(selected_movie, data, movie_titles, top_n=10):
    # Prepare DataFrame
    df = pd.DataFrame(data)
    df['title'] = [movie[0] for movie in movie_titles]
    df['imdb_link'] = [movie[2] for movie in movie_titles]
    
    # Combine features (e.g., genres, plot, etc.) into a single string
    df['combined_features'] = df.apply(lambda x: ' '.join(map(str, x)), axis=1)
    
    # Vectorize the combined features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the selected movie
    try:
        selected_index = df[df['title'] == selected_movie].index[0]
    except IndexError:
        return ["Movie not found in the dataset. Please try another one."]
    
    # Get similarity scores for all movies with the selected movie
    similarity_scores = list(enumerate(cosine_sim[selected_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies
    similar_movies = similarity_scores[1:top_n+1]
    recommendations = [(df.iloc[i[0]]['title'], df.iloc[i[0]]['imdb_link'], i[1]) for i in similar_movies]
    return recommendations

st.set_page_config(
    page_title="Movie Recommender System",
)

def run():
    img1 = Image.open('./meta/logo.jpg')
    img1 = img1.resize((250, 250), )
    st.image(img1, use_column_width=False)
    st.title("Movie Recommender System")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "IMDB 5000 Movie Dataset"</h4>''',
                unsafe_allow_html=True)
    
    # Load Data
    data, movie_titles = load_data()
    
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
              'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    movies = [title[0] for title in movie_titles]
    category = ['--Select--', 'Movie based', 'Genre based']
    cat_op = st.selectbox('Select Recommendation Type', category)
    
    if cat_op == category[0]:
        st.warning('Please select Recommendation Type!!')
    elif cat_op == category[1]:
        select_movie = st.selectbox('Select movie: (Recommendation will be based on this selection)',
                                    ['--Select--'] + movies)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* Fetching a Movie Posters will take a time."</h4>''',
            unsafe_allow_html=True)
        if dec == 'No':
            if select_movie == '--Select--':
                st.warning('Please select Movie!!')
            else:
                no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
                table = content_based_recommender(select_movie, data, movie_titles, no_of_reco)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                for movie, link, similarity in table:
                    c += 1
                    director, cast, story, total_rat = get_movie_info(link)
                    st.markdown(f"({c}) [**{movie}**]({link})")
                    st.markdown(director)
                    st.markdown(cast)
                    st.markdown(story)
                    st.markdown(f"Similarity Score: {similarity:.2f}")
        else:
            if select_movie == '--Select--':
                st.warning('Please select Movie!!')
            else:
                no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
                table = content_based_recommender(select_movie, data, movie_titles, no_of_reco)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                for movie, link, similarity in table:
                    c += 1
                    st.markdown(f"({c}) [**{movie}**]({link})")
                    movie_poster_fetcher(link)
                    director, cast, story, total_rat = get_movie_info(link)
                    st.markdown(director)
                    st.markdown(cast)
                    st.markdown(story)
                    st.markdown(f"Similarity Score: {similarity:.2f}")

run()
