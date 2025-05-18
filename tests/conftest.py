import pytest
import pandas as pd
import numpy as np
import os

@pytest.fixture
def sample_ratings_df():
    #Create a sample ratings DataFrame for testing
    return pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'movieId': [1, 2, 3, 1, 3, 1, 2, 3, 4],
        'rating': [5.0, 3.0, 4.0, 3.0, 5.0, 4.0, 3.5, 1.0, 5.0],
        'timestamp': [1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000007, 1000008]
    })

@pytest.fixture
def sample_movies_df():
    #Create a sample movies DataFrame for testing
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Test Movie 1', 'Test Movie 2', 'Test Movie 3', 'Test Movie 4', 'Test Movie 5'],
        'genres': ['Action|Adventure', 'Comedy|Romance', 'Drama|Thriller', 'Action|Sci-Fi', 'Comedy']
    })

@pytest.fixture
def sample_tags_df():
    #Create a sample tags DataFrame for testing
    return pd.DataFrame({
        'userId': [1, 1, 2, 3, 3],
        'movieId': [1, 2, 3, 1, 4],
        'tag': ['exciting', 'funny', 'intense', 'great effects', 'thought-provoking'],
        'timestamp': [1000000, 1000001, 1000002, 1000003, 1000004]
    })

@pytest.fixture
def mock_model():
    #Create a mock model for testing recommendations
    class MockModel:
        def predict(self, inputs, verbose=0):
            # Simple mock prediction - just return ratings between 3-5
            user_idxs, movie_idxs = inputs
            return np.array([4.0 + (movie_idx % 2) * 0.5 for movie_idx in movie_idxs]).reshape(-1, 1)
    
    return MockModel()

@pytest.fixture
def preprocessed_data(sample_ratings_df, sample_movies_df):
    #Create preprocessed data for testing
    from components.preprocess.preprocess import preprocess_data
    
    return preprocess_data(sample_ratings_df, sample_movies_df)