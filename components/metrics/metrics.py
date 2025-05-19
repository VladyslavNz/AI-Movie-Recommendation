import numpy as np
from sklearn.metrics import ndcg_score

def calculate_metrics(model, user_idxs, movie_idxs, true_ratings):

    predictions = model.predict([user_idxs, movie_idxs], verbose=0).flatten()
    
    mse = np.mean((predictions - true_ratings) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_ratings))
    
    threshold = 4.0
    relevant_mask = true_ratings >= threshold
    precision_recall_metrics = {}
    unique_users = np.unique(user_idxs)
    k_values = [5, 10]
    
    avg_precision = {k: 0.0 for k in k_values}
    avg_recall = {k: 0.0 for k in k_values}
    avg_ndcg = {k: 0.0 for k in k_values}
    
    user_count = 0
    
    for user in unique_users:
        user_mask = user_idxs == user
        if np.sum(user_mask) < max(k_values):
            continue
            
        user_count += 1
        user_movies = movie_idxs[user_mask]
        user_true_ratings = true_ratings[user_mask]
        user_predictions = predictions[user_mask]
    
        sorted_indices = np.argsort(user_predictions)[::-1]  
        
        user_relevance = (user_true_ratings >= threshold).astype(int)
        
        for k in k_values:
            if len(sorted_indices) < k:
                continue
                
            top_k_indices = sorted_indices[:k]
            
            # Calculate Precision@K
            precision_k = np.mean(user_relevance[top_k_indices])
            avg_precision[k] += precision_k
            
            # Calculate Recall@K
            total_relevant = np.sum(user_relevance)
            if total_relevant > 0:
                recall_k = np.sum(user_relevance[top_k_indices]) / total_relevant
                avg_recall[k] += recall_k
            
            # Calculate nDCG@K
            try:
                # Reshape for ndcg_score function requirements
                y_true = user_true_ratings.reshape(1, -1)
                y_scores = user_predictions.reshape(1, -1)
                ndcg = ndcg_score(y_true, y_scores, k=k)
                avg_ndcg[k] += ndcg
            except Exception:
                relevance_sorted = user_relevance[sorted_indices]
                dcg = np.sum(relevance_sorted[:k] / np.log2(np.arange(2, k + 2)))
                ideal_sorted = np.sort(user_relevance)[::-1]
                idcg = np.sum(ideal_sorted[:k] / np.log2(np.arange(2, k + 2)))
                ndcg = dcg / idcg if idcg > 0 else 0
                avg_ndcg[k] += ndcg
    
    if user_count > 0:
        for k in k_values:
            precision_recall_metrics[f'Precision@{k}'] = avg_precision[k] / user_count
            precision_recall_metrics[f'Recall@{k}'] = avg_recall[k] / user_count
            precision_recall_metrics[f'nDCG@{k}'] = avg_ndcg[k] / user_count
    
    metrics = {
        'MSE': mse, 
        'RMSE': rmse, 
        'MAE': mae,
        **precision_recall_metrics
    }
    
    return metrics