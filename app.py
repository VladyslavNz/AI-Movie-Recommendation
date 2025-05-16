import streamlit as st
import plotly.express as px
from components.context.context import initialize_all, initialize_model
from components.recommend.recommend import (get_movie_recommendations, explain_recommendations,create_user_preference_chart, classify_recommendation,get_interactive_explanation)
import os


st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🎬 AI Movie Recommender System")

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
            with st.container():
                # divider
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
