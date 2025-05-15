import os
import matplotlib.pyplot as plt
from tensorflow import keras

def train_model(model, train_df, val_df):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.00001)
    history = model.fit(
        [train_df['user_idx'], train_df['movie_idx']],
        train_df['rating'],
        batch_size=64,
        epochs=10,
        validation_data=([val_df['user_idx'], val_df['movie_idx']], val_df['rating']),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    return history

def plot_history(history, save_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()