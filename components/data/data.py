import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from config import get_dataset_path, DATASETS_DIR

def download_movielens_dataset():
    print("Downloading MovieLens dataset...")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    # Use the config paths
    ratings_path = get_dataset_path('ratings.csv')
    movies_path = get_dataset_path('movies.csv')
    tags_path = get_dataset_path('tags.csv')
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(DATASETS_DIR)
        extracted_dir = os.path.join(DATASETS_DIR, "ml-latest-small")
        if os.path.exists(os.path.join(extracted_dir, "ratings.csv")):
            os.rename(os.path.join(extracted_dir, "ratings.csv"), ratings_path)
        if os.path.exists(os.path.join(extracted_dir, "movies.csv")):
            os.rename(os.path.join(extracted_dir, "movies.csv"), movies_path)
        if os.path.exists(os.path.join(extracted_dir, "tags.csv")):
            os.rename(os.path.join(extracted_dir, "tags.csv"), tags_path)
        print("Dataset downloaded and extracted successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_datasets():
    ratings_path = get_dataset_path('ratings.csv')
    movies_path = get_dataset_path('movies.csv')
    tags_path = get_dataset_path('tags.csv')
    
    if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
        print("Dataset files not found. Attempting to download...")
        if not download_movielens_dataset():
            print("Error: Failed to download dataset files.")
            exit(1)
    
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    # Add tags dataframe if available
    tags_df = None
    if os.path.exists(tags_path):
        tags_df = pd.read_csv(tags_path)
    
    return ratings_df, movies_df, tags_df

def extract_genres(genres_str):
    return genres_str.split('|')