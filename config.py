import os
import pathlib

# Define the project root directory
ROOT_DIR = pathlib.Path(__file__).parent.absolute()

# Define paths for various directories
DATASETS_DIR = os.path.join(ROOT_DIR, "Datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
IMAGES_DIR = os.path.join(ROOT_DIR, "images")

# Model and image file names
MODEL_FILENAME = "movie_recommender_model.keras"
HISTORY_IMAGE_FILENAME = "training_history.png"

def ensure_dir_exists(directory):
    """Make sure the specified directory exists"""
    os.makedirs(directory, exist_ok=True)
    return directory

def get_model_path():
    """Return the full path to the model file"""
    ensure_dir_exists(MODELS_DIR)
    return os.path.join(MODELS_DIR, MODEL_FILENAME)

def get_history_image_path():
    """Return the full path to the training history image"""
    ensure_dir_exists(IMAGES_DIR)
    return os.path.join(IMAGES_DIR, HISTORY_IMAGE_FILENAME)

def get_dataset_path(filename):
    """Return the full path to a dataset file"""
    ensure_dir_exists(DATASETS_DIR)
    return os.path.join(DATASETS_DIR, filename)