import pytest
import os
import pandas as pd
from components.data.data import extract_genres, load_datasets
from config import get_dataset_path

def test_extract_genres():
    #Test genre extraction from genre string
    genres_str = "Action|Adventure|Sci-Fi"
    genres = extract_genres(genres_str)
    assert genres == ["Action", "Adventure", "Sci-Fi"]
    
    single_genre = "Comedy"
    genres = extract_genres(single_genre)
    assert genres == ["Comedy"]

def test_load_datasets_with_existing_files(monkeypatch, tmp_path):
    # Test loading datasets when files exist
    temp_dir = tmp_path / "Datasets"
    temp_dir.mkdir()
    
    ratings_csv = temp_dir / "ratings.csv"
    movies_csv = temp_dir / "movies.csv"
    tags_csv = temp_dir / "tags.csv"
    
    pd.DataFrame({'userId': [1], 'movieId': [1], 'rating': [4.0], 'timestamp': [1000000]}).to_csv(ratings_csv, index=False)
    pd.DataFrame({'movieId': [1], 'title': ['Test Movie'], 'genres': ['Action']}).to_csv(movies_csv, index=False)
    pd.DataFrame({'userId': [1], 'movieId': [1], 'tag': ['great'], 'timestamp': [1000000]}).to_csv(tags_csv, index=False)
    
    def mock_get_dataset_path(filename):
        return str(temp_dir / filename)
    
    monkeypatch.setattr('components.data.data.get_dataset_path', mock_get_dataset_path)
    
    ratings_df, movies_df, tags_df = load_datasets()
    
    assert not ratings_df.empty
    assert not movies_df.empty
    assert not tags_df.empty
    assert 'userId' in ratings_df.columns
    assert 'title' in movies_df.columns
    assert 'tag' in tags_df.columns

def test_preprocess_data(sample_ratings_df, sample_movies_df):
    #Test the data preprocessing function
    from components.preprocess.preprocess import preprocess_data
    
    result = preprocess_data(sample_ratings_df, sample_movies_df)
    
    train_df, val_df, movie_id_to_idx, user_id_to_idx, num_users, num_movies, num_genres, all_genres = result
    
    assert len(train_df) + len(val_df) == len(sample_ratings_df)
    
    assert len(movie_id_to_idx) == len(sample_movies_df['movieId'].unique())
    assert len(user_id_to_idx) == len(sample_ratings_df['userId'].unique())
    
    assert num_users == len(sample_ratings_df['userId'].unique())
    assert num_movies == len(sample_movies_df['movieId'].unique())
    
    expected_genres = {'Action', 'Adventure', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Thriller'}
    assert set(all_genres) == expected_genres
    assert num_genres == len(expected_genres)
    
    for genre in all_genres:
        assert genre in sample_movies_df.columns