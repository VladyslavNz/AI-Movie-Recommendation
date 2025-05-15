import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_data(ratings_df, movies_df):
    all_genres = set()
    for genres in movies_df['genres']:
        all_genres.update(genres.split('|'))
    if '(no genres listed)' in all_genres:
        all_genres.remove('(no genres listed)')
    for genre in all_genres:
        movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
    movie_id_to_idx = {id: i for i, id in enumerate(movies_df['movieId'].unique())}
    user_id_to_idx = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
    num_users = len(user_id_to_idx)
    num_movies = len(movie_id_to_idx)
    num_genres = len(all_genres)
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_to_idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_to_idx)
    train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    return (train_df, val_df, movie_id_to_idx, user_id_to_idx, num_users, num_movies, num_genres, all_genres)