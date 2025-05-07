import os
from tensorflow import keras
from components.data.data import load_datasets
from components.preprocess.preprocess import preprocess_data
from components.model.model import build_model
from components.train.train import train_model, plot_history
from components.metrics.metrics import calculate_metrics


# Paths
model_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'movie_recommender_model.keras')
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Data loading and preparation
ratings_df, movies_df = load_datasets()
train_df, val_df, movie_id_to_idx, user_id_to_idx, num_users, num_movies, num_genres, all_genres = preprocess_data(ratings_df, movies_df)

embedding_dim = 50
hidden_layers = [256, 128, 64]

# Model loading or building
if os.path.exists(model_save_path):
    print(f"Loading model from {model_save_path}...")
    model = keras.models.load_model(model_save_path)
else:
    print("Training new model...")
    model = build_model(num_users, num_movies, embedding_dim, hidden_layers)
    model.summary()
    # Training
    history = train_model(model, train_df, val_df)
    history_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'images', 'training_history.png')
    os.makedirs(os.path.dirname(history_img_path), exist_ok=True)
    plot_history(history, history_img_path)
    # Model saving
    model.save(model_save_path)
    print(f"\nModel saved successfully to {model_save_path}!")

# Export all context variables
__all__ = [
    "model", "ratings_df", "movies_df", "train_df", "val_df",
    "movie_id_to_idx", "user_id_to_idx", "num_users", "num_movies", "num_genres", "all_genres"
]