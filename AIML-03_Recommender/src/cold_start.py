"""
Cold-Start Handling Engine.
Step 6 of AIML-03: Handle cold-start users with popularity and content-based fallbacks.
Provides:
1. Global Bayesian-Smoothed Popularity Fallback
2. Content-Based Genre-Filtered Fallback
3. Item-to-Item Semantic Similarity Fallback
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class ColdStartHandler:
    """
    Handles recommendations for unobserved users or items with zero historical interactions.
    """
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.meta_df = None
        self.item_feats_df = None
        self.top_popular_cache = []
        self._load_metadata()

    def _load_metadata(self):
        meta_path = os.path.join(self.data_dir, "movies_metadata.parquet")
        item_feat_path = os.path.join(self.data_dir, "item_features.parquet")
        
        if os.path.exists(meta_path):
            self.meta_df = pd.read_parquet(meta_path)
        if os.path.exists(item_feat_path):
            self.item_feats_df = pd.read_parquet(item_feat_path)
            
        self._compute_bayesian_popularity()

    def _compute_bayesian_popularity(self, min_ratings: int = 5, global_prior: float = 3.53):
        """
        Bayesian Smoothed Wilson/IMDb Weighted Rating:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
        """
        if self.item_feats_df is None or len(self.item_feats_df) == 0:
            from src.data.dataset import CURATED_MOVIES
            self.top_popular_cache = [
                {"movieId": mid, "title": m["title"], "genres": m["genres"], "score": 4.5}
                for mid, m in CURATED_MOVIES.items()
            ]
            return

        df = self.item_feats_df.copy()
        v = df["item_interaction_count"]
        R = df["item_avg_rating"]
        m = min_ratings
        C = global_prior
        
        df["bayesian_score"] = (v / (v + m)) * R + (m / (v + m)) * C
        df_sorted = df.sort_values("bayesian_score", ascending=False)
        
        records = []
        for _, row in df_sorted.head(100).iterrows():
            records.append({
                "movieId": str(row["movieId"]),
                "title": row.get("title", f"Movie #{row['movieId']}"),
                "genres": row.get("genres", "Drama"),
                "score": float(row["bayesian_score"])
            })
        self.top_popular_cache = records

    def get_popularity_fallback(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Fallback 1: Globally popular, highly-rated movies."""
        return self.top_popular_cache[:top_k]

    def get_content_based_fallback(self, preferred_genres: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Fallback 2: Content-based genre filtering.
        Finds movies matching any of the user's selected genres, ranked by popularity.
        """
        if not preferred_genres:
            return self.get_popularity_fallback(top_k)
            
        pref_set = set(g.strip().lower() for g in preferred_genres)
        matched = []
        
        for item in self.top_popular_cache:
            item_genres = set(g.lower() for g in item["genres"].split("|"))
            overlap = len(pref_set.intersection(item_genres))
            if overlap > 0:
                matched_item = dict(item)
                matched_item["genre_match_count"] = overlap
                matched.append(matched_item)
                
        matched.sort(key=lambda x: (x.get("genre_match_count", 0), x.get("score", 0)), reverse=True)
        return matched[:top_k] if matched else self.get_popularity_fallback(top_k)


# Singleton
COLD_START_HANDLER = ColdStartHandler()


def get_cold_start_recommendations(
    preferred_genres: Optional[List[str]] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Helper function to fetch cold-start recommendations."""
    if preferred_genres and len(preferred_genres) > 0:
        return COLD_START_HANDLER.get_content_based_fallback(preferred_genres, top_k)
    return COLD_START_HANDLER.get_popularity_fallback(top_k)
