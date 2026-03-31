import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import pickle

# Configuration
DATA_DIR = r"C:\Users\Krishna\OneDrive\Desktop\movie recommendation system\data"
MODELS_DIR = r"C:\Users\Krishna\OneDrive\Desktop\movie recommendation system\models"

class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.similarity = None
        
    def load_data(self):
        movies = pd.read_csv(os.path.join(DATA_DIR, 'movie_dataset.csv'))
        credits = pd.read_csv(os.path.join(DATA_DIR, 'movie_credits.csv'))
        
        # Merge datasets
        self.movies = movies.merge(credits, on='title')
        return self.movies

    def convert_ast(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert_cast(self, obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    def fetch_director(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    def preprocess(self):
        # Select relevant columns
        self.movies = self.movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        self.movies.dropna(inplace=True)
        
        # Clean columns
        self.movies['genres'] = self.movies['genres'].apply(self.convert_ast)
        self.movies['keywords'] = self.movies['keywords'].apply(self.convert_ast)
        self.movies['cast'] = self.movies['cast'].apply(self.convert_cast)
        self.movies['crew'] = self.movies['crew'].apply(self.fetch_director)
        
        # Remove spaces
        self.movies['genres'] = self.movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['keywords'] = self.movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['cast'] = self.movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['crew'] = self.movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
        
        # Convert overview to list
        self.movies['overview'] = self.movies['overview'].apply(lambda x: x.split())
        
        # Create tags
        self.movies['tags'] = self.movies['overview'] + self.movies['genres'] + self.movies['keywords'] + self.movies['cast'] + self.movies['crew']
        
        # Convert tags to string
        new_df = self.movies[['movie_id', 'title', 'tags']]
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
        new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
        
        return new_df

    def train(self, df):
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        vector = tfidf.fit_transform(df['tags']).toarray()
        self.similarity = cosine_similarity(vector)
        
        # Save models
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            
        with open(os.path.join(MODELS_DIR, 'similarity.pkl'), 'wb') as f:
            pickle.dump(self.similarity, f)
        
        return self.similarity

    def recommend(self, movie_title, df, similarity):
        try:
            movie_index = df[df['title'] == movie_title].index[0]
            distances = similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
            
            recommended_movies = []
            for i in movies_list:
                recommended_movies.append(df.iloc[i[0]].title)
            return recommended_movies
        except:
            return []

if __name__ == "__main__":
    recommender = MovieRecommender()
    raw_data = recommender.load_data()
    processed_df = recommender.preprocess()
    
    # Save the cleaned dataset for reference
    processed_df.to_csv(os.path.join(DATA_DIR, 'cleaned_movie_dataset.csv'), index=False)
    
    similarity = recommender.train(processed_df)
