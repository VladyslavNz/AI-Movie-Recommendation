import streamlit as st
import plotly.express as px
from components.context.context import initialize_all, initialize_model
from components.recommend.recommend import (get_movie_recommendations, explain_recommendations, 
                                           create_user_preference_chart, classify_recommendation, 
                                           get_interactive_explanation, extract_year_from_title)
from components.metrics.metrics import calculate_metrics
import os
import pandas as pd
from config import get_history_image_path

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
val_df = context['val_df']

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

# Model metrics section
col1, col2 = st.columns([1, 1])
with col1:
    with st.expander("🔍 Model Metrics", expanded=False):
        with st.spinner("Calculating metrics..."):
            metrics = calculate_metrics(
                model, 
                val_df['user_idx'].values, 
                val_df['movie_idx'].values, 
                val_df['rating'].values
            )
            
            # Group metrics by type
            regression_metrics = {k: v for k, v in metrics.items() if k in ['MSE', 'RMSE', 'MAE']}
            ranking_metrics = {k: v for k, v in metrics.items() if k not in ['MSE', 'RMSE', 'MAE']}
            
            # Display regression metrics
            st.markdown("### Error Metrics")
            reg_metrics_df = pd.DataFrame({
                'Metric': list(regression_metrics.keys()),
                'Value': list(regression_metrics.values())
            })
            st.table(reg_metrics_df.set_index('Metric'))
            
           # Display ranking metrics
            st.markdown("### Ranking Metrics")
            ranking_metrics_df = pd.DataFrame({
                'Metric': list(ranking_metrics.keys()),
                'Value': list(ranking_metrics.values())
            })
            st.table(ranking_metrics_df.set_index('Metric'))
            
            # Add a toggle for metrics explanation instead of nested expander
            if st.checkbox("ℹ️ Show metrics explanation", value=False):
                st.markdown("""
                ### Error Metrics
                - **MSE (Mean Squared Error)**: Average of squared differences between predictions and actual ratings. Lower is better.
                - **RMSE (Root Mean Squared Error)**: Square root of MSE, giving error in the same units as the ratings. Lower is better.
                - **MAE (Mean Absolute Error)**: Average of absolute differences between predictions and actual ratings. Lower is better.
                
                ### Ranking Metrics
                - **Precision@K**: Proportion of recommended items in the top-K that are relevant (rated ≥4.0). Higher is better.
                - **Recall@K**: Proportion of relevant items that are in the top-K recommendations. Higher is better.
                - **nDCG@K**: Normalized Discounted Cumulative Gain at K. Measures ranking quality considering both relevance and position. Higher is better.
                """)

with col2:
    if st.button("🔁 Retrain model"):
        confirm_retraining()

if st.session_state.retraining_complete:
    st.success("✅ Model retrained and saved!")
    st.image(get_history_image_path(), caption="📈 Training History", use_container_width=True)
    st.session_state.retraining_complete = False

with st.expander("📈 View Neural Network Training History", expanded=False):
    st.image(get_history_image_path(), caption="Neural Network Training Metrics", use_container_width=True)

user_id = st.selectbox("Select a user ID:", unique_user_ids)

with st.expander("📊 View User Preferences", expanded=False):
    st.write("This chart shows your genre preferences based on your rating history:")
    preference_chart = create_user_preference_chart(user_id, ratings_df, movies_df)
    st.plotly_chart(preference_chart, use_container_width=True)
    st.info("📌 Higher bars indicate genres you tend to rate highly. The number on each bar shows how many movies of that genre you've watched.")

top_n = st.slider("How many movies to recommend?", min_value=1, max_value=20, value=10)
selected_mode = "detailed"

# filters section
with st.expander("Filter Recommendations", expanded=False):
    col1, col2 = st.columns(2)
    all_genres = set()
    for genres_str in movies_df['genres'].str.split('|'):
        valid_genres = [genre for genre in genres_str if genre != "(no genres listed)"]
        all_genres.update(valid_genres)
        
    all_genres = sorted(list(all_genres))
    
    years = []
    for title in movies_df['title']:
        import re
        match = re.search(r'\((\d{4})\)', title)
        if match:
            years.append(int(match.group(1)))
    
    min_year = min(years) if years else 1900
    max_year = max(years) if years else 2023
    
    with col1:
        selected_genres = st.multiselect(
            "Filter by genres (leave empty for all):",
            options=all_genres,
            default=None
        )
    
    with col2:
        year_range = st.slider(
            "Filter by year range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

if st.button("🔍 Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        buffer_factor = 3  # Get 3x as many recommendations to have room for filtering
        initial_recommendations = get_movie_recommendations(
            user_id,
            top_n=top_n * buffer_factor,
            model=model,
            movies_df=movies_df,
            ratings_df=ratings_df,
            movie_id_to_idx=movie_id_to_idx,
            user_id_to_idx=user_id_to_idx,
        )
        
        if initial_recommendations is not None and not initial_recommendations.empty:
            filtered_recommendations = initial_recommendations.copy()
            if selected_genres:
                genre_mask = filtered_recommendations['genres'].apply(
                    lambda x: any(genre in x.split('|') for genre in selected_genres)
                )
                filtered_recommendations = filtered_recommendations[genre_mask]
            
            # Filter by year range
            if year_range != (min_year, max_year):
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