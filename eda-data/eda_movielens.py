import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = "/Users/thiennguyen/Library/CloudStorage/GoogleDrive-nguyenvmthien@gmail.com/My Drive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02/src/data/movielens"

def load_data():
    # Load users
    users_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.dat'), sep='::', engine='python', names=users_cols, encoding='ISO-8859-1')
    
    # Load movies
    movies_cols = ['MovieID', 'Title', 'Genres']
    movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.dat'), sep='::', engine='python', names=movies_cols, encoding='ISO-8859-1')
    
    # Load ratings
    ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.dat'), sep='::', engine='python', names=ratings_cols, encoding='ISO-8859-1')
    
    return users, movies, ratings

def perform_eda(users, movies, ratings):
    print("--- MovieLens 1M EDA ---")
    print(f"Number of users: {users['UserID'].nunique()}")
    print(f"Number of movies: {movies['MovieID'].nunique()}")
    print(f"Number of ratings: {len(ratings)}")
    
    # Sparsity
    num_users = users['UserID'].nunique()
    num_items = movies['MovieID'].nunique()
    num_ratings = len(ratings)
    sparsity = 1.0 - (num_ratings / (num_users * num_items))
    print(f"Sparsity: {sparsity:.4f}")
    
    # Rating distribution
    print("\nRating Distribution:")
    print(ratings['Rating'].value_counts(normalize=True).sort_index())
    
    # Users per movie
    users_per_movie = ratings.groupby('MovieID')['UserID'].count()
    print("\nUsers per Movie Statistics:")
    print(users_per_movie.describe())
    
    # Movies per user
    movies_per_user = ratings.groupby('UserID')['MovieID'].count()
    print("\nMovies per User Statistics:")
    print(movies_per_user.describe())
    
    # Top 10 most rated movies
    print("\nTop 10 Most Rated Movies:")
    top_movies = ratings.groupby('MovieID').size().sort_values(ascending=False).head(10)
    top_movies_titles = movies[movies['MovieID'].isin(top_movies.index)][['MovieID', 'Title']]
    print(top_movies_titles)

if __name__ == "__main__":
    users, movies, ratings = load_data()
    perform_eda(users, movies, ratings)
