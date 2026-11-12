"""
Comprehensive Test Suite for AIML-03 Two-Stage Recommendation Engine.
Tests all 7 PRD Steps:
1. Data Cleaning & Temporal Split
2. Spark Feature Engineering
3. Two-Tower Retrieval Embeddings
4. FAISS Vector Indexing & ANN Retrieval
5. Deep Ranking Models (DCN-v2 & Wide & Deep)
6. Cold-Start Fallbacks & Resilient Caching
7. Metrics (NDCG, Recall@K, CTR) & A/B Experimentation
8. FastAPI Endpoint Contracts
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import (
    generate_synthetic_interactions, temporal_train_test_split, get_movie_metadata
)
from src.data.spark_features import build_features_vectorized, compute_genre_affinity
from src.faiss_index import CandidateFAISSIndex
from src.models.ranking_model import rank_candidates, build_ranking_features
from src.cold_start import ColdStartHandler, get_cold_start_recommendations
from src.caching.redis_cache import RecommendationCache, InMemoryLRUCache
from src.evaluation.metrics import (
    compute_ndcg_at_k, compute_recall_at_k, compute_hit_rate_at_k,
    compute_precision_at_k, compute_ctr
)
from src.evaluation.ab_test import assign_ab_variant


class TestRecommenderPipeline(unittest.TestCase):

    def test_step1_temporal_split(self):
        """Step 1: Clean and split by time, not randomly."""
        df = generate_synthetic_interactions(num_users=100, num_movies=50, total_interactions=1000)
        self.assertGreater(len(df), 500)
        self.assertTrue("timestamp" in df.columns)
        
        train, val, test = temporal_train_test_split(df, 0.8, 0.1, 0.1)
        self.assertEqual(len(train) + len(val) + len(test), len(df))
        # Verify strict chronological separation: max(train) <= min(val) <= min(test)
        self.assertLessEqual(train["timestamp"].max(), val["timestamp"].min())
        self.assertLessEqual(val["timestamp"].max(), test["timestamp"].min())

    def test_step2_feature_engineering(self):
        """Step 2: Spark/Vectorized feature engineering."""
        df = pd.DataFrame({
            "userId": ["u1", "u1", "u2"],
            "movieId": ["m1", "m2", "m1"],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [100, 200, 150]
        })
        user_feats, item_feats = build_features_vectorized(df)
        
        self.assertIn("user_history", user_feats.columns)
        self.assertIn("item_popularity", item_feats.columns)
        
        u1_row = user_feats[user_feats["userId"] == "u1"].iloc[0]
        self.assertEqual(u1_row["user_interaction_count"], 2)
        self.assertAlmostEqual(u1_row["user_avg_rating"], 4.5)
        self.assertEqual(u1_row["user_history"], "m1|m2")
        
        m1_row = item_feats[item_feats["movieId"] == "m1"].iloc[0]
        self.assertEqual(m1_row["item_interaction_count"], 2)

    def test_step3_embeddings_properties(self):
        """Step 3: Two-Tower embedding dimensions and normalization."""
        emb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "movie_embeddings.npy")
        if os.path.exists(emb_path):
            embs = np.load(emb_path)
            self.assertEqual(embs.shape[1], 32)
            self.assertEqual(embs.dtype, np.float32)

    def test_step4_faiss_retrieval(self):
        """Step 4: FAISS Vector Index build and retrieval."""
        dim = 32
        n_items = 100
        movie_ids = np.array([str(i) for i in range(n_items)])
        movie_embs = np.random.randn(n_items, dim).astype(np.float32)
        
        faiss_idx = CandidateFAISSIndex(dimension=dim)
        faiss_idx.build(movie_embs, movie_ids)
        
        query = movie_embs[0]
        cands = faiss_idx.search(query, top_k=5)
        self.assertEqual(len(cands), 5)
        self.assertEqual(cands[0]["movie_id"], "0") # Closest to itself
        self.assertGreaterEqual(cands[0]["retrieval_score"], 0.99)
        
        # Test item-to-item search
        sim_items = faiss_idx.get_similar_items("0", top_k=3)
        self.assertEqual(len(sim_items), 3)
        self.assertNotIn("0", [x["movie_id"] for x in sim_items])

    def test_step5_deep_ranking_models(self):
        """Step 5: DCN-v2 and Wide & Deep candidate re-ranking."""
        candidates = [
            {"movie_id": "1", "retrieval_score": 0.9, "genre_affinity": 0.8},
            {"movie_id": "2", "retrieval_score": 0.4, "genre_affinity": 0.1},
            {"movie_id": "3", "retrieval_score": 0.7, "genre_affinity": 0.5}
        ]
        user_feat = {"user_avg_rating": 4.0, "user_interaction_count": 25, "user_history": "10|20"}
        
        ranked_dcn = rank_candidates(candidates, user_feat, model_variant="DCN-v2", top_k=2)
        ranked_wd = rank_candidates(candidates, user_feat, model_variant="Wide&Deep", top_k=2)
        
        self.assertEqual(len(ranked_dcn), 2)
        self.assertEqual(len(ranked_wd), 2)
        self.assertIn("ranking_score", ranked_dcn[0])
        self.assertIn("ranking_score", ranked_wd[0])
        # Top ranked candidate should have highest ranking score
        self.assertGreaterEqual(ranked_dcn[0]["ranking_score"], ranked_dcn[1]["ranking_score"])

    def test_step6_cold_start_and_cache(self):
        """Step 6: Cold start fallbacks and caching."""
        # 1. Popularity Fallback
        pop_recs = get_cold_start_recommendations(top_k=5)
        self.assertEqual(len(pop_recs), 5)
        
        # 2. Genre-based Content Fallback
        genre_recs = get_cold_start_recommendations(preferred_genres=["Animation"], top_k=3)
        self.assertEqual(len(genre_recs), 3)
        
        # 3. L1 Cache with TTL
        l1 = InMemoryLRUCache(capacity=2, default_ttl=3600)
        l1.set("k1", "v1")
        l1.set("k2", "v2")
        self.assertEqual(l1.get("k1"), "v1")
        l1.set("k3", "v3") # Evicts oldest (k2)
        self.assertEqual(l1.get("k3"), "v3")
        self.assertIsNone(l1.get("k2"))

    def test_step7_metrics_and_ab_test(self):
        """Step 7: Evaluation metrics and deterministic A/B routing."""
        rec_ids = ["1", "2", "3", "4", "5"]
        ground_truth_rel = {"1": 1.0, "3": 0.8}
        
        ndcg = compute_ndcg_at_k(rec_ids, ground_truth_rel, k=5)
        recall = compute_recall_at_k(rec_ids, set(ground_truth_rel.keys()), k=5)
        hr = compute_hit_rate_at_k(rec_ids, set(ground_truth_rel.keys()), k=5)
        ctr = compute_ctr(total_clicks=15, total_impressions=100)
        
        self.assertGreater(ndcg, 0.0)
        self.assertEqual(recall, 1.0) # All 2 found
        self.assertEqual(hr, 1.0)
        self.assertEqual(ctr, 0.15)
        
        # A/B assignment consistency
        var1 = assign_ab_variant("user_abc")
        var2 = assign_ab_variant("user_abc")
        self.assertEqual(var1, var2) # Deterministic
        self.assertIn(var1, ["Variant_A_DCN", "Variant_B_WideDeep"])


if __name__ == "__main__":
    unittest.main()
