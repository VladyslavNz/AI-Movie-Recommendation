import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import io

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths to datasets
dataset_path = os.path.join(os.path.dirname(__file__), 'Datasets')
ratings_path = os.path.join(dataset_path, 'ratings.csv')
movies_path = os.path.join(dataset_path, 'movies.csv')


# Function to download and extract MovieLens dataset
def download_movielens_dataset():
    print("Downloading MovieLens dataset...")
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)

    # MovieLens 100K dataset URL
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    try:
        # Download the dataset
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dataset_path)

        # MovieLens small dataset is extracted to ml-latest-small directory
        # Move files to the expected location
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


print("Loading datasets...")

# Check if dataset files exist, download if they don't
if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
    print("Dataset files not found. Attempting to download...")
    if not download_movielens_dataset():
        print("Error: Failed to download dataset files.")
        exit(1)

# Load datasets
try:
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    print("Datasets loaded successfully!")
except FileNotFoundError:
    print(
        "Error: Dataset files not found. Please make sure 'ratings.csv' and 'movies.csv' exist in the Datasets folder.")
    exit(1)

# Display basic information about the datasets
print("\nRatings dataset info:")
print(f"Total ratings: {len(ratings_df)}")
print(ratings_df.head())

print("\nMovies dataset info:")
print(f"Total movies: {len(movies_df)}")
print(movies_df.head())

# Preprocess the data
print("\nPreprocessing data...")


# Extract genres and create one-hot encoding
def extract_genres(genres_str):
    genres_list = genres_str.split('|')
    return genres_list


# Get all unique genres
all_genres = set()
for genres in movies_df['genres']:
    all_genres.update(extract_genres(genres))

# Remove potential 'no genres listed' or similar entries
if '(no genres listed)' in all_genres:
    all_genres.remove('(no genres listed)')

# Create genre features
for genre in all_genres:
    movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x else 0)

# Get a mapping of movie IDs to their indices
movie_id_to_idx = {id: i for i, id in enumerate(movies_df['movieId'].unique())}
user_id_to_idx = {id: i for i, id in enumerate(ratings_df['userId'].unique())}

# Number of users and movies for embedding dimensions
num_users = len(user_id_to_idx)
num_movies = len(movie_id_to_idx)
num_genres = len(all_genres)

print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Number of genres: {num_genres}")

# Prepare training data
ratings_df['user_idx'] = ratings_df['userId'].map(user_id_to_idx)
ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_to_idx)

# Split data into training and validation sets
train_df, val_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Build the neural network model
print("\nBuilding neural network model...")

# Define model parameters
embedding_dim = 50  # Dimension of embedding vectors
hidden_layers = [256, 128, 64]  # Hidden layer dimensions

# User input
user_input = keras.Input(shape=(1,), name='user_input')
user_embedding = layers.Embedding(
    num_users,
    embedding_dim,
    embeddings_regularizer=keras.regularizers.l2(0.01),  # Add L2 regularization
    name='user_embedding'
)(user_input)
user_vec = layers.Flatten(name='flatten_user')(user_embedding)

# Movie input
movie_input = keras.Input(shape=(1,), name='movie_input')
movie_embedding = layers.Embedding(
    num_movies,
    embedding_dim,
    embeddings_regularizer=keras.regularizers.l2(0.01),  # Add L2 regularization
    name='movie_embedding'
)(movie_input)
movie_vec = layers.Flatten(name='flatten_movie')(movie_embedding)

# Add bias terms (explicit modeling of user and movie biases)
user_bias = layers.Embedding(
    num_users,
    1,
    embeddings_regularizer=keras.regularizers.l2(0.01),
    name='user_bias'
)(user_input)
user_bias = layers.Flatten(name='flatten_user_bias')(user_bias)

movie_bias = layers.Embedding(
    num_movies,
    1,
    embeddings_regularizer=keras.regularizers.l2(0.01),
    name='movie_bias'
)(movie_input)
movie_bias = layers.Flatten(name='flatten_movie_bias')(movie_bias)

# Merge user and movie embeddings
concat = layers.Concatenate()([user_vec, movie_vec])

# Add hidden layers
dense = concat
for i, units in enumerate(hidden_layers):
    dense = layers.Dense(
        units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01),  # Add L2 regularization
        name=f'dense_{i}'
    )(dense)
    dense = layers.Dropout(0.3)(dense)  # Slightly increase dropout

# Combine deep component with bias terms
output_bias = layers.Add()([user_bias, movie_bias])
deep_output = layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01), name='deep_output')(dense)
output = layers.Add(name='output')([deep_output, output_bias])

# Create model
model = keras.Model(
    inputs=[user_input, movie_input],
    outputs=output
)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error'
)

# Print model summary
model.summary()

# Add early stopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Add learning rate scheduler for better convergence
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=0.00001
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    [train_df['user_idx'], train_df['movie_idx']],
    train_df['rating'],
    batch_size=64,
    epochs=10,
    validation_data=(
        [val_df['user_idx'], val_df['movie_idx']],
        val_df['rating']
    ),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), 'training_history.png'))
plt.close()  # Close the plot instead of showing it (to avoid hanging)


# Function to calculate evaluation metrics
def calculate_metrics(model, user_idxs, movie_idxs, true_ratings):
    predictions = model.predict([user_idxs, movie_idxs], verbose=0).flatten()

    # Calculate MSE and RMSE
    mse = np.mean((predictions - true_ratings) ** 2)
    rmse = np.sqrt(mse)

    # Calculate MAE
    mae = np.mean(np.abs(predictions - true_ratings))

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }


# Calculate metrics on validation set
val_metrics = calculate_metrics(
    model,
    val_df['user_idx'].values,
    val_df['movie_idx'].values,
    val_df['rating'].values
)
print("\nValidation metrics:")
for metric, value in val_metrics.items():
    print(f"{metric}: {value:.4f}")


# Function to get movie recommendations for a user
def get_movie_recommendations(user_id, top_n=10):
    """
    Get personalized movie recommendations for a specific user.

    Parameters:
    user_id (int): The ID of the user to get recommendations for
    top_n (int): Number of recommendations to return

    Returns:
    pandas.DataFrame: DataFrame containing recommended movies and their predicted ratings
    """
    if user_id not in user_id_to_idx:
        print(f"User ID {user_id} not found in the dataset.")
        return None

    user_idx = user_id_to_idx[user_id]

    # Get movies the user hasn't rated yet
    user_rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    unrated_movies = movies_df[~movies_df['movieId'].isin(user_rated_movies)]

    if len(unrated_movies) == 0:
        print(f"User {user_id} has rated all available movies.")
        return None

    # Prepare prediction data
    user_idxs = np.array([user_idx] * len(unrated_movies))
    movie_idxs = np.array([movie_id_to_idx.get(id, 0) for id in unrated_movies['movieId']])

    # Predict ratings
    predicted_ratings = model.predict([user_idxs, movie_idxs], verbose=0)

    # Add predictions to the DataFrame
    unrated_movies = unrated_movies.copy()
    unrated_movies['predicted_rating'] = predicted_ratings

    # Sort by predicted rating and get top N recommendations
    recommendations = unrated_movies.sort_values('predicted_rating', ascending=False).head(top_n)

    return recommendations[['movieId', 'title', 'genres', 'predicted_rating']]


# Save the model with proper file extension
model_save_path = os.path.join(os.path.dirname(__file__), 'movie_recommender_model.keras')
model.save(model_save_path)
print(f"\nModel saved successfully to {model_save_path}!")

# Example: Get movie recommendations for a specific user
user_id_example = ratings_df['userId'].iloc[0]  # Get the first user as an example
print(f"\nMovie recommendations for User ID {user_id_example}:")
try:
    recommendations = get_movie_recommendations(user_id_example, top_n=10)
    if recommendations is not None:
        print(recommendations)
except Exception as e:
    print(f"Error getting recommendations: {e}")


# Function to explain recommendations
def explain_recommendations(user_id, movie_id):
    """
    Provide a simple explanation for why a movie was recommended to a user.

    Parameters:
    user_id (int): The ID of the user
    movie_id (int): The ID of the recommended movie

    Returns:
    str: Explanation of the recommendation
    """
    # Get the movie information
    movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
    movie_title = movie_info['title']
    movie_genres = movie_info['genres'].split('|')

    # Find similar movies the user has rated highly
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    user_high_ratings = user_ratings[user_ratings['rating'] >= 4.0]

    # Get genres of highly rated movies
    high_rated_movies = movies_df[movies_df['movieId'].isin(user_high_ratings['movieId'])]
    user_genre_preferences = {}

    for _, movie in high_rated_movies.iterrows():
        for genre in movie['genres'].split('|'):
            if genre in user_genre_preferences:
                user_genre_preferences[genre] += 1
            else:
                user_genre_preferences[genre] = 1

    # Find common genres
    common_genres = [genre for genre in movie_genres if genre in user_genre_preferences]

    if common_genres:
        return f"'{movie_title}' was recommended because you liked movies in the {', '.join(common_genres)} genre(s)."
    else:
        return f"'{movie_title}' was recommended based on your overall rating patterns."

# Get recommendations for user with ID 1 (or other ID from the dataset)
specific_user_id = 6
recommendations = get_movie_recommendations(specific_user_id, top_n=10)
print(f"\nРекомендации для пользователя {specific_user_id}:")
print(recommendations)

# Get an explanation for the first recommendation
if recommendations is not None and not recommendations.empty:
    first_movie_id = recommendations.iloc[0]['movieId']
    explanation = explain_recommendations(specific_user_id, first_movie_id)
    print(f"\nОбъяснение рекомендации:")
    print(explanation)

# Provide instructions for using the recommendation system
print("\nTo get movie recommendations for a specific user, use the following code:")
print("recommendations = get_movie_recommendations(user_id, top_n=10)")
print("print(recommendations)")

print("\nTo get an explanation for a recommendation, use:")
print("explanation = explain_recommendations(user_id, movie_id)")
print("print(explanation)")