import os
from tensorflow import keras
from components.data.data import load_datasets
from components.preprocess.preprocess import preprocess_data
from components.model.model import build_model
from components.train.train import train_model, plot_history
from config import get_model_path, get_history_image_path

model = None
ratings_df = None
movies_df = None
tags_df = None 
train_df = None
val_df = None
movie_id_to_idx = None
user_id_to_idx = None
num_users = None
num_movies = None
num_genres = None
all_genres = None

# Configuration
EMBEDDING_DIM = 50
HIDDEN_LAYERS = [256, 128, 64]

def initialize_data():
    #Load and preprocess data, return initialized variables
    global ratings_df, movies_df, tags_df, train_df, val_df, movie_id_to_idx, user_id_to_idx, num_users, num_movies, num_genres, all_genres
    
    # Load data if not already loaded
    if ratings_df is None or movies_df is None:
        ratings_df, movies_df, tags_df = load_datasets()
    
    # Preprocess data
    train_df, val_df, movie_id_to_idx, user_id_to_idx, num_users, num_movies, num_genres, all_genres = preprocess_data(ratings_df, movies_df)
    
    return {
        'ratings_df': ratings_df, 
        'movies_df': movies_df,
        'tags_df': tags_df,
        'train_df': train_df, 
        'val_df': val_df,
        'movie_id_to_idx': movie_id_to_idx,
        'user_id_to_idx': user_id_to_idx,
        'num_users': num_users,
        'num_movies': num_movies,
        'num_genres': num_genres,
        'all_genres': all_genres
    }

def initialize_model(force_retrain=False):
    #Initialize or load the model, train if necessary
    global model
    
    # Make sure data is initialized
    data = initialize_data()
    
    model_path = get_model_path()
    
    # Load existing model if available and not forced to retrain
    if os.path.exists(model_path) and not force_retrain:
        print(f"Loading model from {model_path}...")
        model = keras.models.load_model(model_path)
    else:
        print("Training new model...")
        model = build_model(data['num_users'], data['num_movies'], EMBEDDING_DIM, HIDDEN_LAYERS)
        model.summary()
        
        # Train the model
        history = train_model(model, data['train_df'], data['val_df'])
        
        # Plot and save training history
        history_img_path = get_history_image_path()
        plot_history(history, history_img_path)
        
        # Save the model
        model.save(model_path)
        print(f"\nModel saved successfully to {model_path}!")
    
    return model

def initialize_all(force_retrain=False):
    #Initialize all context variables
    initialize_data()
    initialize_model(force_retrain)
    
    # Return all context variables as a dictionary
    return {
        'model': model,
        'ratings_df': ratings_df,
        'movies_df': movies_df,
        'tags_df': tags_df, 
        'train_df': train_df,
        'val_df': val_df,
        'movie_id_to_idx': movie_id_to_idx,
        'user_id_to_idx': user_id_to_idx,
        'num_users': num_users,
        'num_movies': num_movies,
        'num_genres': num_genres,
        'all_genres': all_genres
    }