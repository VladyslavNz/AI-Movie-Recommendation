import streamlit as st
import pandas as pd
from main import get_movie_recommendations, explain_recommendations  # Импорт из твоего основного кода

st.set_page_config(page_title="AI Movie Recommender", layout="wide")

st.title("🎬 AI Movie Recommender System")

user_id = st.number_input("Enter User ID:", min_value=1, max_value=610, step=1)

if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        try:
            recommendations = get_movie_recommendations(user_id, top_n=10)
            st.success("Top Recommendations:")

            st.dataframe(recommendations[["title", "genres", "predicted_rating"]])

            selected_movie = st.selectbox(
                "Choose a movie to explain why it was recommended:",
                recommendations["title"].tolist()
            )
            if selected_movie:
                movie_id = recommendations[recommendations["title"] == selected_movie]["movieId"].values[0]
                explanation = explain_recommendations(user_id, movie_id)
                st.info(f"🧠 {explanation}")

        except Exception as e:
            st.error(f"Error: {e}")
