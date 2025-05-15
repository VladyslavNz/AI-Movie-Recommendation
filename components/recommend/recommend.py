import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

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

def get_genre_preferences(user_id, ratings_df, movies_df):
    """Analyze user genre preferences with average ratings per genre."""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    # Dictionary to store total ratings and count for each genre
    genre_ratings = {}
    genre_counts = {}
    
    # Collect ratings by genre
    for _, row in user_ratings.iterrows():
        movie_id = row['movieId']
        rating = row['rating']
        
        movie_row = movies_df[movies_df['movieId'] == movie_id]
        if not movie_row.empty:
            genres = movie_row.iloc[0]['genres'].split('|')
            for genre in genres:
                if genre != '(no genres listed)':
                    genre_ratings[genre] = genre_ratings.get(genre, 0) + rating
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Calculate average rating per genre
    genre_avg_ratings = {}
    for genre in genre_ratings:
        if genre_counts[genre] > 0:
            genre_avg_ratings[genre] = genre_ratings[genre] / genre_counts[genre]
    
    # Sort genres by average rating (descending)
    sorted_genres = sorted(genre_avg_ratings.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'avg_ratings': genre_avg_ratings,
        'counts': genre_counts,
        'sorted_preferences': sorted_genres
    }

def get_tag_preferences(user_id, ratings_df, movies_df, tags_df):
    """Analyze user tag preferences using TF-IDF."""
    if tags_df is None:
        return None
    
    # Get movies rated by the user
    user_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    
    # Get tags for those movies
    movie_tags = tags_df[tags_df['movieId'].isin(user_movies)]
    
    # If no tags available, return None
    if len(movie_tags) == 0:
        return None
    
    # Prepare tag counts
    tag_counts = Counter(movie_tags['tag'].str.lower())
    
    # Create tag document for TF-IDF (all tags for user's movies as one document)
    user_tags_document = ' '.join(movie_tags['tag'].str.lower())
    
    # Perform TF-IDF analysis if we have enough data
    if len(user_tags_document) > 0:
        try:
            # Create a corpus with user's tags and a general set of tags
            corpus = [user_tags_document]
            
            # Add a general tag document for comparison
            general_tags = tags_df[~tags_df['movieId'].isin(user_movies)]['tag'].str.lower()
            general_tags_document = ' '.join(general_tags[:5000])  # Limit to prevent memory issues
            if general_tags_document:
                corpus.append(general_tags_document)
            
            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get user document vector (first document in corpus)
            user_vector = tfidf_matrix[0]
            
            # Get top tags by TF-IDF score
            scores = zip(feature_names, user_vector.toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Return both frequency counts and TF-IDF results
            return {
                'tag_counts': tag_counts,
                'tfidf_tags': sorted_scores[:20]  # Top 20 tags by TF-IDF score
            }
        except Exception as e:
            # Fallback to simple frequency analysis if TF-IDF fails
            print(f"TF-IDF analysis failed: {e}")
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            return {
                'tag_counts': tag_counts,
                'tfidf_tags': None,
                'frequency_tags': sorted_tags[:20]
            }
    
    # If no TF-IDF analysis was possible, return frequency analysis
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return {
        'tag_counts': tag_counts,
        'tfidf_tags': None,
        'frequency_tags': sorted_tags[:20]
    }

def find_similar_users(user_id, movie_id, ratings_df, min_common=5):
    """Find users with similar taste profiles who rated the recommended movie."""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    # Get all users who rated the recommended movie
    movie_raters = ratings_df[ratings_df['movieId'] == movie_id]
    
    similar_users = []
    
    # For each user who rated the movie
    for rater_id in movie_raters['userId'].unique():
        if rater_id == user_id:
            continue  # Skip the user themselves
            
        rater_ratings = ratings_df[ratings_df['userId'] == rater_id]
        
        # Find movies both users rated
        common_movies = set(user_ratings['movieId']).intersection(set(rater_ratings['movieId']))
        
        # Only consider users who have rated enough movies in common
        if len(common_movies) >= min_common:
            # Get ratings for common movies
            user_common_ratings = user_ratings[user_ratings['movieId'].isin(common_movies)]
            rater_common_ratings = rater_ratings[rater_ratings['movieId'].isin(common_movies)]
            
            # Sort both by movieId to ensure alignment
            user_common_ratings = user_common_ratings.sort_values('movieId')
            rater_common_ratings = rater_common_ratings.sort_values('movieId')
            
            # Check for variance in ratings - correlation can't be calculated if ratings are all identical
            user_std = np.std(user_common_ratings['rating'].values)
            rater_std = np.std(rater_common_ratings['rating'].values)
            
            if user_std > 0 and rater_std > 0:
                # Calculate similarity (correlation between ratings)
                try:
                    similarity = np.corrcoef(
                        user_common_ratings['rating'].values,
                        rater_common_ratings['rating'].values
                    )[0,1]
                    
                    # Only include if similarity is positive and not NaN
                    if not np.isnan(similarity) and similarity > 0.3:
                        rater_movie_rating = movie_raters[movie_raters['userId'] == rater_id]['rating'].iloc[0]
                        similar_users.append({
                            'userId': rater_id,
                            'similarity': similarity,
                            'movie_rating': rater_movie_rating,
                            'common_movies': len(common_movies)
                        })
                except Exception as e:
                    # Skip this user if correlation calculation fails
                    print(f"Error calculating similarity: {e}")
                    continue
    
    # Sort by similarity
    similar_users = sorted(similar_users, key=lambda x: x['similarity'], reverse=True)
    
    return similar_users[:5]  # Return top 5 similar users

def explain_recommendations(user_id, movie_id, ratings_df, movies_df, tags_df=None):
    """Enhanced explanation for movie recommendations."""
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        return "No explanation available."
    
    movie_title = movie_row.iloc[0]['title']
    movie_genres = movie_row.iloc[0]['genres'].split('|')
    movie_year = extract_year_from_title(movie_title)
    
    # List to store different explanation components
    explanations = []
    confidence_signals = 0  # Track how many positive signals we have
    
    # 1. Genre-based explanation (improved)
    genre_prefs = get_genre_preferences(user_id, ratings_df, movies_df)
    
    if genre_prefs and genre_prefs['sorted_preferences']:
        # Check if any of the movie's genres are among the user's top preferences
        user_top_genres = [g[0] for g in genre_prefs['sorted_preferences'][:3]]
        common_top_genres = [g for g in movie_genres if g in user_top_genres]
        
        if common_top_genres:
            genre_ratings = [f"{g} ({genre_prefs['avg_ratings'][g]:.1f}⭐)" for g in common_top_genres]
            explanations.append(f"You tend to enjoy {', '.join(genre_ratings)} movies")
            confidence_signals += len(common_top_genres)
    
    # 2. Tag-based explanation
    if tags_df is not None:
        # Get movie tags
        movie_tags = tags_df[tags_df['movieId'] == movie_id]['tag'].str.lower().tolist()
        
        if movie_tags:
            # Get user tag preferences
            user_tag_prefs = get_tag_preferences(user_id, ratings_df, movies_df, tags_df)
            
            if user_tag_prefs:
                # Use TF-IDF results if available, otherwise frequency
                if user_tag_prefs['tfidf_tags']:
                    user_important_tags = [tag for tag, _ in user_tag_prefs['tfidf_tags'][:10]]
                elif 'frequency_tags' in user_tag_prefs and user_tag_prefs['frequency_tags']:
                    user_important_tags = [tag for tag, _ in user_tag_prefs['frequency_tags'][:10]]
                else:
                    user_important_tags = []
                
                # Find tags in common
                common_tags = [tag for tag in movie_tags if tag.lower() in user_important_tags]
                
                if common_tags:
                    unique_tags = list(set(common_tags))[:3]  # Limit to 3 tags for readability
                    explanations.append(f"This movie has tags you often look for: {', '.join(unique_tags)}")
                    confidence_signals += len(unique_tags)
    
    # 3. Similar users explanation
    similar_users = find_similar_users(user_id, movie_id, ratings_df)
    
    if similar_users:
        # Calculate average rating from similar users
        avg_rating = sum(u['movie_rating'] for u in similar_users) / len(similar_users)
        count = len(similar_users)
        
        # Find the most similar user for a more personal touch
        most_similar = max(similar_users, key=lambda x: x['similarity'])
        
        if most_similar['similarity'] > 0.6 and most_similar['movie_rating'] > 4.0:
            explanations.append(f"A user with very similar taste gave this movie {most_similar['movie_rating']}⭐")
            confidence_signals += 2
        else:
            explanations.append(f"{count} users with similar taste rated this movie {avg_rating:.1f}⭐ on average")
            confidence_signals += 1 if avg_rating > 3.5 else 0
    
    # 4. Year/era preference analysis
    if movie_year:
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        # Create an explicit copy of the filtered DataFrame
        user_movies = movies_df[movies_df['movieId'].isin(user_ratings['movieId'])].copy()
        
        # Now we can safely add a column
        user_movies['year'] = user_movies['title'].apply(extract_year_from_title)
        
        # Count movies from same decade
        decade = (movie_year // 10) * 10
        decade_movies = user_movies[user_movies['year'] >= decade]
        decade_movies = decade_movies[decade_movies['year'] < decade + 10]
        
        if len(decade_movies) >= 3:
            decade_avg = user_ratings[user_ratings['movieId'].isin(decade_movies['movieId'])]['rating'].mean()
            if decade_avg > 3.8:
                explanations.append(f"You've enjoyed movies from the {decade}s ({decade_avg:.1f}⭐ average)")
                confidence_signals += 1
    
    # 5. Find director/actor preferences if possible
    # This would require additional data not currently available in the dataset
    
    # 6. Generic fallback explanation if no other explanations
    if not explanations:
        # Check if the user has rated movies of these genres before
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        user_movies = movies_df[movies_df['movieId'].isin(user_ratings['movieId'])]
        
        has_rated_similar_genres = False
        for genre in movie_genres:
            if any(user_movies['genres'].str.contains(genre)):
                has_rated_similar_genres = True
                break
        
        if has_rated_similar_genres:
            explanations.append(f"This matches your viewing history")
        else:
            explanations.append(f"This might expand your viewing preferences")
    
    # 7. Add confidence level based on accumulated signals
    confidence_prefix = ""
    if confidence_signals >= 5:
        confidence_prefix = "🌟 Strong match: "
    elif confidence_signals >= 3:
        confidence_prefix = "✅ Good match: "
    
    # Combine all explanations with confidence prefix
    return confidence_prefix + " • ".join(explanations)

def extract_year_from_title(title):
    """Extract year from movie title if available (format: "Title (YYYY)")"""
    import re
    match = re.search(r'\((\d{4})\)$', title)
    if match:
        return int(match.group(1))
    return None

def create_user_preference_chart(user_id, ratings_df, movies_df):
    """
    Creates a visualization of user genre preferences with ratings.
    
    Args:
        user_id: The user ID to analyze
        ratings_df: DataFrame with ratings data
        movies_df: DataFrame with movies data
        
    Returns:
        A plotly figure object showing genre preferences
    """
    # Get user genre preferences
    user_genre_prefs = get_genre_preferences(user_id, ratings_df, movies_df)
    
    if not user_genre_prefs or not user_genre_prefs['sorted_preferences']:
        # Create empty chart with message if no preferences
        fig = px.bar(x=["No data"], y=[0], 
                    title=f"User {user_id} has no genre preferences data")
        fig.add_annotation(text="No genre preferences data available", 
                          showarrow=False, font_size=16)
        return fig
    
    # Create dataframe for plotting
    genres = [g[0] for g in user_genre_prefs['sorted_preferences']]
    ratings = [g[1] for g in user_genre_prefs['sorted_preferences']]
    counts = [user_genre_prefs['counts'][g] for g in genres]
    
    plot_data = pd.DataFrame({
        'Genre': genres,
        'Average Rating': ratings,
        'Movies Watched': counts
    })
    
    # Create bar chart with hover info
    fig = px.bar(
        plot_data, 
        x='Genre', 
        y='Average Rating',
        title=f"Genre Preferences for User {user_id}",
        color='Average Rating',
        color_continuous_scale='RdYlGn',  # Red to Yellow to Green scale
        hover_data=['Movies Watched'],
        text='Movies Watched'
    )
    
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title="Average Rating (⭐)",
        yaxis_range=[0, 5],
        xaxis_tickangle=-45
    )
    
    # Add horizontal line at rating 3.5 for reference
    fig.add_hline(y=3.5, line_dash="dash", line_color="gray", 
                 annotation_text="Average rating threshold", 
                 annotation_position="bottom right")
    
    return fig

def classify_recommendation(user_id, movie_id, ratings_df, movies_df):
    """
    Classifies a recommendation as either similar to user favorites or something new.
    
    Args:
        user_id: The user ID 
        movie_id: The movie ID to classify
        ratings_df: DataFrame with ratings data
        movies_df: DataFrame with movies data
        
    Returns:
        A classification string and emoji indicator
    """
    # Get user preferences
    user_genre_prefs = get_genre_preferences(user_id, ratings_df, movies_df)
    
    if not user_genre_prefs or not user_genre_prefs['sorted_preferences']:
        return "🆕 Something new to try"
    
    # Get top genres for the user
    top_genres = [g[0] for g in user_genre_prefs['sorted_preferences'][:3]]
    
    # Get movie genres
    movie_row = movies_df[movies_df['movieId'] == movie_id]
    if movie_row.empty:
        return "⁉️ Unknown classification"
    
    movie_genres = movie_row.iloc[0]['genres'].split('|')
    common = set(movie_genres).intersection(set(top_genres))
    
    # Classify based on genre overlap
    if len(common) > 1:
        return f"👍 Similar to your favorites ({', '.join(common)})"
    elif len(common) == 1:
        return f"🔄 Mix of familiar and new ({common.pop()})"
    else:
        return "🆕 Something new to try"