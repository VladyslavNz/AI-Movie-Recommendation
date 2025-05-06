import numpy as np

def get_movie_recommendations(user_id, top_n, model, movies_df, ratings_df, movie_id_to_idx, user_id_to_idx):
    user_idx = user_id_to_idx.get(user_id, None)
    if user_idx is None:
        return None
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unrated_movies = movies_df[~movies_df['movieId'].isin(rated_movies)]
    user_idxs = np.full(len(unrated_movies), user_idx)
    movie_idxs = np.array([movie_id_to_idx.get(id, 0) for id in unrated_movies['movieId']])
    predicted_ratings = model.predict([user_idxs, movie_idxs], verbose=0)
    unrated_movies = unrated_movies.copy()
    unrated_movies['predicted_rating'] = predicted_ratings
    recommendations = unrated_movies.sort_values('predicted_rating', ascending=False).head(top_n)
    return recommendations[['movieId', 'title', 'genres', 'predicted_rating']]

def explain_recommendations(user_id, movie_id, ratings_df, movies_df):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        return "No explanation available."
    movie_title = movie_row.iloc[0]['title']
    movie_genres = movie_row.iloc[0]['genres'].split('|')
    user_genre_preferences = {}
    for _, row in user_ratings.iterrows():
        genres = movies_df[movies_df['movieId'] == row['movieId']]['genres']
        if not genres.empty:
            for genre in genres.iloc[0].split('|'):
                user_genre_preferences[genre] = user_genre_preferences.get(genre, 0) + 1
    common_genres = [genre for genre in movie_genres if genre in user_genre_preferences]
    if common_genres:
        return f"'{movie_title}' was recommended because you liked movies in the {', '.join(common_genres)} genre(s)."
    else:
        return f"'{movie_title}' was recommended based on your overall rating patterns."