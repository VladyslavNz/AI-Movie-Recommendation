import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io

# Get the project root (assumes this script is always in components/data/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

dataset_path = os.path.join(project_root, 'Datasets')
ratings_path = os.path.join(dataset_path, 'ratings.csv')
movies_path = os.path.join(dataset_path, 'movies.csv')

def download_movielens_dataset():
    print("Downloading MovieLens dataset...")
    os.makedirs(dataset_path, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    try:
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dataset_path)
        extracted_dir = os.path.join(dataset_path, "ml-latest-small")
        if os.path.exists(os.path.join(extracted_dir, "ratings.csv")):
            os.rename(os.path.join(extracted_dir, "ratings.csv"), ratings_path)
        if os.path.exists(os.path.join(extracted_dir, "movies.csv")):
            os.rename(os.path.join(extracted_dir, "movies.csv"), movies_path)
        print("Dataset downloaded and extracted successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_datasets():
    if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
        print("Dataset files not found. Attempting to download...")
        if not download_movielens_dataset():
            print("Error: Failed to download dataset files.")
            exit(1)
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    return ratings_df, movies_df

def extract_genres(genres_str):
    return genres_str.split('|')