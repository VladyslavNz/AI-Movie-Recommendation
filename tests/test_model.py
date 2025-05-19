import pytest
import os
import numpy as np
import tensorflow as tf
from config import get_model_path

def test_model_exists():
    #Test that the model file exists
    model_path = get_model_path()
    assert os.path.exists(model_path)

@pytest.mark.skipif(not os.path.exists(get_model_path()), 
                    reason="Model file not available")
def test_model_loading():
    #Test that the model can be loaded
    try:
        model = tf.keras.models.load_model(get_model_path())
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

@pytest.mark.skipif(not os.path.exists(get_model_path()), 
                    reason="Model file not available")
def test_model_prediction():
    #Test that the model can make predictions
    model = tf.keras.models.load_model(get_model_path())
    
    user_idx = np.array([0, 1, 2])
    movie_idx = np.array([0, 1, 2])
    
    predictions = model.predict([user_idx, movie_idx], verbose=0)
    
    # Check if predictions have expected shape and values
    assert predictions.shape == (3, 1)
    assert np.all((predictions >= 0) & (predictions <= 5))