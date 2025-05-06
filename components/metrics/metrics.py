import numpy as np

def calculate_metrics(model, user_idxs, movie_idxs, true_ratings):
    predictions = model.predict([user_idxs, movie_idxs], verbose=0).flatten()
    mse = np.mean((predictions - true_ratings) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_ratings))
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}