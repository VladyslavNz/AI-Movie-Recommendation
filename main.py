from components.context.context import initialize_all
from components.metrics.metrics import calculate_metrics
from components.recommend.recommend import get_movie_recommendations, explain_recommendations

# Initialize all context variables
context = initialize_all()

# Extract context variables
model = context['model']
ratings_df = context['ratings_df']
movies_df = context['movies_df']
val_df = context['val_df']
movie_id_to_idx = context['movie_id_to_idx']
user_id_to_idx = context['user_id_to_idx']

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

print(f"\nRecommendations for user {user_id_example}:")
print(recommendations)

if recommendations is not None and not recommendations.empty:
    first_movie_id = recommendations.iloc[0]['movieId']
    explanation = explain_recommendations(user_id_example, first_movie_id, ratings_df, movies_df)
    print(explanation)