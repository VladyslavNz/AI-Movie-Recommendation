import streamlit as st
import plotly.express as px
from components.context.context import initialize_all, initialize_model
from components.recommend.recommend import (get_movie_recommendations, explain_recommendations, 
                                           create_user_preference_chart, classify_recommendation, 
                                           get_interactive_explanation, extract_year_from_title)
import os
import pandas as pd

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender System")

if 'retraining_complete' not in st.session_state:
    st.session_state.retraining_complete = False
    st.session_state.retrained_model = None

# Initialize context (lazy loading)
@st.cache_resource
def get_context():
    return initialize_all()

# Get context variables
context = get_context()
model = context['model']
ratings_df = context['ratings_df']
movies_df = context['movies_df']
tags_df = context['tags_df']
movie_id_to_idx = context['movie_id_to_idx']
user_id_to_idx = context['user_id_to_idx']

if st.session_state.retrained_model is not None:
    model = st.session_state.retrained_model

# Dialog for model retraining
@st.dialog("Confirm Model Retraining")
def confirm_retraining():
    st.warning("⚠️ **This will retrain the model**")
    st.write("Model retraining may take several minutes and will overwrite the current model. Are you sure you want to proceed?")

    if st.button("Yes, retrain model"):
        with st.spinner("Training the model, please wait..."):
            # Force model retraining
            st.session_state.retrained_model = initialize_model(force_retrain=True)
            st.session_state.retraining_complete = True
            st.rerun()

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
selected_mode = "detailed"

# Add filters section
with st.expander("Filter Recommendations", expanded=False):
    col1, col2 = st.columns(2)
    all_genres = set()
    for genres_str in movies_df['genres'].str.split('|'):
        valid_genres = [genre for genre in genres_str if genre != "(no genres listed)"]
        all_genres.update(valid_genres)
        
    all_genres = sorted(list(all_genres))
    
    # Get years range from dataset
    years = []
    for title in movies_df['title']:
        import re
        match = re.search(r'\((\d{4})\)', title)
        if match:
            years.append(int(match.group(1)))
    
    min_year = min(years) if years else 1900
    max_year = max(years) if years else 2023
    
    with col1:
        # Genre filter (multi-select)
        selected_genres = st.multiselect(
            "Filter by genres (leave empty for all):",
            options=all_genres,
            default=None
        )
    
    with col2:
        # Year range filter
        year_range = st.slider(
            "Filter by year range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

if st.button("🔁 Retrain model"):
    confirm_retraining()

if st.session_state.retraining_complete:
    st.success("✅ Model retrained and saved!")
    history_img_path = os.path.join(os.path.dirname(__file__), 'images', 'training_history.png')
    st.image(history_img_path, caption="📈 Training History", use_container_width=True)
    st.session_state.retraining_complete = False

# Generate recommendations
if st.button("🔍 Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        # Get more recommendations than needed to allow for filtering
        buffer_factor = 3  # Get 3x as many recommendations to have room for filtering
        initial_recommendations = get_movie_recommendations(
            user_id,
            top_n=top_n * buffer_factor,  # Get more recommendations initially
            model=model,
            movies_df=movies_df,
            ratings_df=ratings_df,
            movie_id_to_idx=movie_id_to_idx,
            user_id_to_idx=user_id_to_idx,
        )
        
        if initial_recommendations is not None and not initial_recommendations.empty:
            # Apply filters if specified
            filtered_recommendations = initial_recommendations.copy()
            # Filter by selected genres
            if selected_genres:
                # Create mask for movies that contain ANY of the selected genres
                genre_mask = filtered_recommendations['genres'].apply(
                    lambda x: any(genre in x.split('|') for genre in selected_genres)
                )
                filtered_recommendations = filtered_recommendations[genre_mask]
            
            # Filter by year range
            if year_range != (min_year, max_year):
                # Extract years and filter
                year_mask = filtered_recommendations['title'].apply(
                    lambda x: extract_year_from_title(x) is not None and 
                              year_range[0] <= extract_year_from_title(x) <= year_range[1]
                )
                filtered_recommendations = filtered_recommendations[year_mask]
            
            recommendations = filtered_recommendations.head(top_n)
            
            if recommendations.empty and not initial_recommendations.empty:
                st.warning("No movies match your filter criteria. Try adjusting your filters.")
            elif recommendations is not None and not recommendations.empty:
                # Show filtering info if any filters were applied
                filter_applied = selected_genres or year_range != (min_year, max_year)
                if filter_applied:
                    filters_description = []
                    if selected_genres:
                        filters_description.append(f"genres: {', '.join(selected_genres)}")
                    if year_range != (min_year, max_year):
                        filters_description.append(f"years: {year_range[0]}-{year_range[1]}")
                    
                    st.success(f"Recommended {len(recommendations)} movies for user {user_id} matching filters: {', '.join(filters_description)}")
                else:
                    st.success(f"Recommended {len(recommendations)} movies for user {user_id}")
                
                for idx, row in recommendations.iterrows():
                    with st.container():
                        if idx > 0:
                            st.markdown("---")
                        st.markdown(f"### {row['title']} ({row['predicted_rating']:.2f} ⭐)")
                        st.markdown(f"**Genres**: {row['genres']}")
                        classification = classify_recommendation(user_id, row['movieId'], ratings_df, movies_df)
                        if "New to you" in classification:
                            st.markdown(f"🆕 **{classification}**")
                        else:
                            st.markdown(f"🔄 **{classification}**")
                        
                        # Get explanation
                        explanation = get_interactive_explanation(user_id, row['movieId'], ratings_df, movies_df, tags_df, mode=selected_mode)
                        
                        if explanation.startswith("🌟 Strong match:"):
                            confidence_text = "🌟 **Strong match**"
                            explanation_content = explanation[len("🌟 Strong match:"):]
                            confidence_color = "green"
                        elif explanation.startswith("✅ Good match:"):
                            confidence_text = "✅ **Good match**"
                            explanation_content = explanation[len("✅ Good match:"):]
                            confidence_color = "orange"
                        else:
                            confidence_text = "💡 **Match reason**"
                            explanation_content = explanation
                            confidence_color = "gray"
                        
                        st.markdown(f":{confidence_color}[{confidence_text}]")
                        
                        if "\n\n" in explanation_content:
                            basic_part, detailed_part = explanation_content.split("\n\n", 1)
                            st.markdown(basic_part)
                            with st.expander("👁️ See detailed explanation"):
                                st.markdown(detailed_part)
                        else:
                            st.markdown(explanation_content)
        else:
            st.warning("No recommendations found for this user.")