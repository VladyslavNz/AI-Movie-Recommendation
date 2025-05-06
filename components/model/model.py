from tensorflow import keras
from tensorflow.keras import layers

def build_model(num_users, num_movies, embedding_dim, hidden_layers):
    user_input = keras.Input(shape=(1,), name='user_input')
    user_embedding = layers.Embedding(
        num_users, embedding_dim, embeddings_regularizer=keras.regularizers.l2(0.01), name='user_embedding'
    )(user_input)
    user_vec = layers.Flatten(name='flatten_user')(user_embedding)
    movie_input = keras.Input(shape=(1,), name='movie_input')
    movie_embedding = layers.Embedding(
        num_movies, embedding_dim, embeddings_regularizer=keras.regularizers.l2(0.01), name='movie_embedding'
    )(movie_input)
    movie_vec = layers.Flatten(name='flatten_movie')(movie_embedding)
    user_bias = layers.Embedding(num_users, 1, embeddings_regularizer=keras.regularizers.l2(0.01), name='user_bias')(user_input)
    user_bias = layers.Flatten(name='flatten_user_bias')(user_bias)
    movie_bias = layers.Embedding(num_movies, 1, embeddings_regularizer=keras.regularizers.l2(0.01), name='movie_bias')(movie_input)
    movie_bias = layers.Flatten(name='flatten_movie_bias')(movie_bias)
    concat = layers.Concatenate()([user_vec, movie_vec])
    dense = concat
    for i, units in enumerate(hidden_layers):
        dense = layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name=f'dense_{i}')(dense)
        dense = layers.Dropout(0.3)(dense)
    output_bias = layers.Add()([user_bias, movie_bias])
    deep_output = layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01), name='deep_output')(dense)
    output = layers.Add(name='output')([deep_output, output_bias])
    model = keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model