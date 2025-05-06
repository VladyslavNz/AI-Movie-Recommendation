import streamlit as st
import pandas as pd
from main import (
    get_movie_recommendations,
    explain_recommendations,
    ratings_df,
    movies_df,
    model,
    movie_id_to_idx,
    user_id_to_idx,
)


# Page settings
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender System")

# Get all unique user IDs
unique_user_ids = sorted(ratings_df['userId'].unique())

# User ID selection
user_id = st.selectbox("Select a user ID:", unique_user_ids)

# Number of recommendations
top_n = st.slider("How many movies to recommend?", min_value=1, max_value=20, value=10)

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

        # Display each recommended movie
        for idx, row in recommendations.iterrows():
            st.subheader(f"{row['title']} ({row['predicted_rating']:.2f} ⭐)")
            st.text(f"Genres: {row['genres']}")
            explanation = explain_recommendations(user_id, row['movieId'], ratings_df, movies_df)
            st.caption(f"📌 {explanation}")
            st.markdown("---")
    else:
        st.warning("No recommendations found for this user.")
