import streamlit as st
import plotly.express as px
from components.context.context import initialize_all, initialize_model
from components.recommend.recommend import (get_movie_recommendations, explain_recommendations,
                                           create_user_preference_chart, classify_recommendation)
import os

# Page settings
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender System")

# Initialize context (with lazy loading)
@st.cache_resource
def get_context():
    return initialize_all()

# Get context variables
context = get_context()
model = context['model']
ratings_df = context['ratings_df']
movies_df = context['movies_df']
tags_df = context['tags_df']  # New addition
movie_id_to_idx = context['movie_id_to_idx']
user_id_to_idx = context['user_id_to_idx']

# Get all unique user IDs
unique_user_ids = sorted(ratings_df['userId'].unique())

# User ID selection
user_id = st.selectbox("Select a user ID:", unique_user_ids)

# Add a section for user preferences visualization
with st.expander("📊 View User Preferences", expanded=False):
    st.write("This chart shows your genre preferences based on your rating history:")
    preference_chart = create_user_preference_chart(user_id, ratings_df, movies_df)
    st.plotly_chart(preference_chart, use_container_width=True)
    st.info("📌 Higher bars indicate genres you tend to rate highly. The number on each bar shows how many movies of that genre you've watched.")

# Number of recommendations
top_n = st.slider("How many movies to recommend?", min_value=1, max_value=20, value=10)

# Retrain model button
if st.button("🔁 Retrain model"):
    with st.spinner("Training the model, please wait..."):
        # Force model retraining
        model = initialize_model(force_retrain=True)
        
        # Display training history
        history_img_path = os.path.join(os.path.dirname(__file__), 'images', 'training_history.png')
        st.image(history_img_path, caption="📈 Training History", use_container_width=True)
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
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"{row['title']} ({row['predicted_rating']:.2f} ⭐)")
                st.text(f"Genres: {row['genres']}")
                
                # Add the classification tag
                classification = classify_recommendation(user_id, row['movieId'], ratings_df, movies_df)
                st.info(classification)
                
                explanation = explain_recommendations(user_id, row['movieId'], ratings_df, movies_df, tags_df)
                st.caption(f"📌 {explanation}")
            
            # Add a button to see similar movies in the future
            with col2:
                if st.button(f"Similar to #{row['movieId']}", key=f"similar_{row['movieId']}"):
                    st.session_state.selected_movie_id = row['movieId']
            
            st.markdown("---")
    else:
        st.warning("No recommendations found for this user.")
