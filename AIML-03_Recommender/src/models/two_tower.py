"""
Two-Tower Candidate Retrieval Model using TensorFlow Recommenders (TFRS).
Step 3 of AIML-03: Train a two-tower retrieval model in TensorFlow Recommenders that creates user and item embeddings.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text, Tuple, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
TRITON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "triton_models")


class UserModel(tf.keras.Model):
    """Query Tower: Maps userId string to 32-dimensional L2-normalized embedding."""
    def __init__(self, unique_user_ids: np.ndarray, embedding_dimension: int = 32):
        super().__init__()
        self.user_lookup = tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None)
        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_user_ids) + 1,
            output_dim=embedding_dimension
        )
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension)
        ])

    def call(self, inputs):
        x = self.user_lookup(inputs)
        x = self.user_embedding(x)
        x = self.dense(x)
        return tf.math.l2_normalize(x, axis=-1)


class MovieModel(tf.keras.Model):
    """Candidate Tower: Maps movieId string to 32-dimensional L2-normalized embedding."""
    def __init__(self, unique_movie_ids: np.ndarray, embedding_dimension: int = 32):
        super().__init__()
        self.movie_lookup = tf.keras.layers.StringLookup(vocabulary=unique_movie_ids, mask_token=None)
        self.movie_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_movie_ids) + 1,
            output_dim=embedding_dimension
        )
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dimension)
        ])

    def call(self, inputs):
        x = self.movie_lookup(inputs)
        x = self.movie_embedding(x)
        x = self.dense(x)
        return tf.math.l2_normalize(x, axis=-1)


class TwoTowerRecommender(tfrs.Model):
    """Complete Two-Tower retrieval model with FactorizedTopK candidate evaluation."""
    def __init__(self, user_model: UserModel, movie_model: MovieModel, candidate_dataset: tf.data.Dataset):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_dataset.batch(128).map(self.movie_model)
            )
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movieId"])
        return self.task(user_embeddings, movie_embeddings)


def train_two_tower_model(
    epochs: int = 3,
    batch_size: int = 512,
    embedding_dim: int = 32,
    data_dir: str = DATA_DIR,
    models_dir: str = MODELS_DIR
):
    """Trains Two-Tower retrieval model on train_ratings.parquet and exports embeddings."""
    os.makedirs(models_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train_ratings.parquet")
    meta_path = os.path.join(data_dir, "movies_metadata.parquet")
    
    if not os.path.exists(train_path):
        from src.data.dataset import prepare_data_pipeline
        prepare_data_pipeline(data_dir)
        
    print("[Two-Tower] Loading training interactions and movie metadata...")
    train_df = pd.read_parquet(train_path)
    
    if os.path.exists(meta_path):
        movies_meta = pd.read_parquet(meta_path)
        unique_movies = movies_meta["movieId"].unique().astype(str)
    else:
        unique_movies = train_df["movieId"].unique().astype(str)
        
    unique_users = train_df["userId"].unique().astype(str)
    
    print(f"[Two-Tower] Vocabulary: {len(unique_users):,} Users | {len(unique_movies):,} Movies")
    
    # TensorFlow Datasets
    interactions_dict = {
        "userId": train_df["userId"].values.astype(str),
        "movieId": train_df["movieId"].values.astype(str)
    }
    interactions_ds = tf.data.Dataset.from_tensor_slices(interactions_dict)
    movies_ds = tf.data.Dataset.from_tensor_slices(unique_movies)
    
    train_ds = interactions_ds.shuffle(100_000).batch(batch_size).cache()
    
    print("[Two-Tower] Initializing User and Movie Towers...")
    user_model = UserModel(unique_users, embedding_dim)
    movie_model = MovieModel(unique_movies, embedding_dim)
    
    model = TwoTowerRecommender(user_model, movie_model, movies_ds)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05))
    
    print(f"[Two-Tower] Training for {epochs} epochs...")
    model.fit(train_ds, epochs=epochs, verbose=1)
    
    print("[Two-Tower] Generating and exporting embeddings...")
    # Generate movie embeddings
    movie_batch_ds = movies_ds.batch(512)
    movie_embeddings = []
    for batch in movie_batch_ds:
        emb = movie_model(batch).numpy()
        movie_embeddings.append(emb)
    movie_embeddings = np.vstack(movie_embeddings).astype(np.float32)
    
    # Generate user embeddings
    users_ds = tf.data.Dataset.from_tensor_slices(unique_users).batch(512)
    user_embeddings = []
    for batch in users_ds:
        emb = user_model(batch).numpy()
        user_embeddings.append(emb)
    user_embeddings = np.vstack(user_embeddings).astype(np.float32)
    
    # Save artifacts
    np.save(os.path.join(models_dir, "unique_users.npy"), unique_users)
    np.save(os.path.join(models_dir, "unique_movies.npy"), unique_movies)
    np.save(os.path.join(models_dir, "user_embeddings.npy"), user_embeddings)
    np.save(os.path.join(models_dir, "movie_embeddings.npy"), movie_embeddings)
    
    print(f"Exported artifacts to {models_dir}:")
    print(f"  - unique_users.npy: {user_embeddings.shape}")
    print(f"  - movie_embeddings.npy: {movie_embeddings.shape}")
    
    return model, user_model, movie_model


if __name__ == "__main__":
    train_two_tower_model(epochs=1)
