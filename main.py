from components.data.data import load_datasets, extract_genres
from components.preprocess.preprocess import preprocess_data
from components.model.model import build_model
from components.train.train import train_model, plot_history
from components.metrics.metrics import calculate_metrics
from components.recommend.recommend import get_movie_recommendations, explain_recommendations
import os
from tensorflow import keras

# Model saving path
model_save_path = os.path.join(os.path.dirname(__file__), 'models', 'movie_recommender_model.keras')
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
    history_img_path = os.path.join(os.path.dirname(__file__), 'images', 'training_history.png')
    os.makedirs(os.path.dirname(history_img_path), exist_ok=True)
    plot_history(history, history_img_path)
    # Model saving
    model.save(model_save_path)
    print(f"\nModel saved successfully to {model_save_path}!")

# Evaluation
val_metrics = calculate_metrics(model, val_df['user_idx'].values, val_df['movie_idx'].values, val_df['rating'].values)
print("\nValidation metrics:")
for metric, value in val_metrics.items():
    print(f"{metric}: {value:.4f}")

# Example of recommendations and explanation
user_id_example = ratings_df['userId'].iloc[0]
recommendations = get_movie_recommendations(
    user_id_example, top_n=10, model=model, movies_df=movies_df, ratings_df=ratings_df,
    movie_id_to_idx=movie_id_to_idx, user_id_to_idx=user_id_to_idx
)
print(recommendations)
if recommendations is not None and not recommendations.empty:
    first_movie_id = recommendations.iloc[0]['movieId']
    explanation = explain_recommendations(user_id_example, first_movie_id, ratings_df, movies_df)
    print(explanation)