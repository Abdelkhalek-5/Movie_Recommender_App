import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, data, feature_column):
        """
        :param data: DataFrame containing movie info
        :param feature_column: column used for similarity (e.g., 'genres' or 'description')
        """
        self.data = data.copy()
        self.feature_column = feature_column
        self.sim_matrix = None
        self.indices = None

    def prepare(self):
        """Vectorize the feature column and compute similarity matrix"""
        self.data[self.feature_column] = self.data[self.feature_column].fillna('').astype(str)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.data[self.feature_column])
        self.sim_matrix = cosine_similarity(tfidf_matrix)
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()

    def recommend(self, title, k=5):
        """Recommend top-k similar movies based on content"""
        if title not in self.indices:
            print("❌ Movie not found!")
            return []
        
        idx = self.indices[title]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.data[['title', self.feature_column]].iloc[movie_indices]
