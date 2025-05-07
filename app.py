import streamlit as st
import os
from components.context.context import (
    model,
    ratings_df,
    movies_df,
    train_df,
    val_df,
    movie_id_to_idx,
    user_id_to_idx,
    num_users,
    num_movies,
    num_genres,
    all_genres,
)
from components.preprocess.preprocess import preprocess_data
from components.model.model import build_model
from components.train.train import train_model, plot_history
from components.recommend.recommend import get_movie_recommendations, explain_recommendations

# Page settings
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender System")

# Get all unique user IDs
unique_user_ids = sorted(ratings_df['userId'].unique())

# User ID selection
user_id = st.selectbox("Select a user ID:", unique_user_ids)

# Number of recommendations
top_n = st.slider("How many movies to recommend?", min_value=1, max_value=20, value=10)

if st.button("🔁 Retrain model"):
    with st.spinner("Training the model, please wait..."):
        # Re-preprocessing data
        train_df, val_df, movie_id_to_idx, user_id_to_idx, num_users, num_movies, num_genres, all_genres = preprocess_data(ratings_df, movies_df)

        # Build and train new model
        embedding_dim = 50
        hidden_layers = [256, 128, 64]
        new_model = build_model(num_users, num_movies, embedding_dim, hidden_layers)
        history = train_model(new_model, train_df, val_df)

        # Save model
        model_save_path = os.path.join(os.path.dirname(__file__), 'models', 'movie_recommender_model.keras')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        new_model.save(model_save_path)

        # Plot training history
        history_img_path = os.path.join(os.path.dirname(__file__), 'images', 'training_history.png')
        plot_history(history, history_img_path)
        st.image(history_img_path, caption="📈 Training History", use_column_width=True)
        st.success("✅ Model retrained and saved!")

# Generate recommendations
if st.button("🔍 Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations = get_movie_recommendations(
            user_id,
            top_n=top_n,
            model=model,
            movies_df=movies_df,
            ratings_df=ratings_df,
            movie_id_to_idx=movie_id_to_idx,
            user_id_to_idx=user_id_to_idx,
        )

    if recommendations is not None and not recommendations.empty:
        st.success(f"Recommended {len(recommendations)} movies for user {user_id}:")
        for idx, row in recommendations.iterrows():
            st.subheader(f"{row['title']} ({row['predicted_rating']:.2f} ⭐)")
            st.text(f"Genres: {row['genres']}")
            explanation = explain_recommendations(user_id, row['movieId'], ratings_df, movies_df)
            st.caption(f"📌 {explanation}")
            st.markdown("---")
    else:
        st.warning("No recommendations found for this user.")
