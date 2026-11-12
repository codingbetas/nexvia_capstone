"""
Online A/B Testing Experimentation Framework.
Step 7 of AIML-03: Run a simple online A/B test comparing two ranking models. Measure NDCG, recall@K, and click-through.
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Ensure root workspace is on python sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.evaluation.metrics import (
    compute_ndcg_at_k, compute_recall_at_k, compute_hit_rate_at_k,
    compute_precision_at_k, compute_ctr, welch_t_test
)
from src.faiss_index import get_default_faiss_index
from src.models.ranking_model import rank_candidates
from src.cold_start import get_cold_start_recommendations

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def assign_ab_variant(user_id: str) -> str:
    """
    Deterministic user bucketing via MD5 hash modulo 2.
    Variant A: DCN-v2 Ranker (Deep & Cross Network)
    Variant B: Wide & Deep Ranker
    """
    hash_val = int(hashlib.md5(user_id.encode("utf-8")).hexdigest(), 16)
    return "Variant_A_DCN" if hash_val % 2 == 0 else "Variant_B_WideDeep"


def simulate_user_click(recommended_mid: str, relevant_dict: Dict[str, float], base_ctr: float = 0.05) -> int:
    """
    Simulates user click behavior:
    If item is in user's future ground-truth test interactions with rating >= 3.5: high click probability (65-80%).
    Otherwise: baseline exploratory click probability (~5%).
    """
    if str(recommended_mid) in relevant_dict:
        rel = relevant_dict[str(recommended_mid)]
        p_click = 0.50 + 0.35 * rel
    else:
        p_click = base_ctr
    return 1 if np.random.rand() < p_click else 0


def run_ab_experiment(
    num_test_users: int = 500,
    top_k: int = 10,
    retrieval_candidates_k: int = 50,
    data_dir: str = DATA_DIR,
    models_dir: str = MODELS_DIR
) -> Dict[str, Any]:
    """
    Executes an end-to-end A/B experiment simulation on temporal test users.
    Compares Variant A (DCN-v2) against Variant B (Wide & Deep).
    """
    print(f"\n{'='*70}\n[A/B Test] Launching Online A/B Test Simulation (N={num_test_users:,} Users)\n{'='*70}")
    
    test_ratings_path = os.path.join(data_dir, "test_ratings.parquet")
    user_feats_path = os.path.join(data_dir, "user_features.parquet")
    item_feats_path = os.path.join(data_dir, "item_features.parquet")
    
    if not os.path.exists(test_ratings_path):
        from src.data.dataset import prepare_data_pipeline
        prepare_data_pipeline(data_dir)
        
    test_df = pd.read_parquet(test_ratings_path)
    user_feats_df = pd.read_parquet(user_feats_path) if os.path.exists(user_feats_path) else pd.DataFrame()
    item_feats_df = pd.read_parquet(item_feats_path) if os.path.exists(item_feats_path) else pd.DataFrame()
    
    user_feats_map = {}
    if len(user_feats_df) > 0:
        user_feats_map = {str(row["userId"]): row.to_dict() for _, row in user_feats_df.iterrows()}
        
    item_feats_map = {}
    if len(item_feats_df) > 0:
        item_feats_map = {str(row["movieId"]): row.to_dict() for _, row in item_feats_df.iterrows()}
        
    faiss_index = get_default_faiss_index()
    
    # Load user embeddings
    user_emb_path = os.path.join(models_dir, "user_embeddings.npy")
    unique_users_path = os.path.join(models_dir, "unique_users.npy")
    
    user_embs = np.load(user_emb_path)
    unique_users = np.load(unique_users_path, allow_pickle=True)
    user_to_idx = {str(uid): i for i, uid in enumerate(unique_users)}
    
    # Group test interactions by user: userId -> {movieId: normalized_relevance}
    test_ground_truth = {}
    for uid, grp in test_df.groupby("userId"):
        # Normalized relevance: (rating - 2.5) / 2.5, clipped at 0
        rels = {}
        for _, row in grp.iterrows():
            rel_score = max(0.0, (row["rating"] - 2.5) / 2.5)
            if rel_score > 0.0:
                rels[str(row["movieId"])] = rel_score
        if rels:
            test_ground_truth[str(uid)] = rels
            
    eval_users = list(test_ground_truth.keys())[:num_test_users]
    if len(eval_users) == 0:
        eval_users = [str(u) for u in unique_users[:num_test_users]]
        test_ground_truth = {u: {"1": 1.0, "296": 1.0, "318": 1.0} for u in eval_users}
        
    metrics = {
        "Variant_A_DCN": {
            "users": 0, "ndcg": [], "recall": [], "hit_rate": [], "precision": [],
            "clicks": 0, "impressions": 0, "latencies_ms": []
        },
        "Variant_B_WideDeep": {
            "users": 0, "ndcg": [], "recall": [], "hit_rate": [], "precision": [],
            "clicks": 0, "impressions": 0, "latencies_ms": []
        }
    }
    
    for uid in eval_users:
        variant = assign_ab_variant(uid)
        model_name = "DCN-v2" if "DCN" in variant else "Wide&Deep"
        
        t0 = time.perf_counter()
        
        # 1. Stage 1: Retrieval
        if uid in user_to_idx:
            u_idx = user_to_idx[uid]
            u_vec = user_embs[u_idx]
            candidates = faiss_index.search(u_vec, top_k=retrieval_candidates_k)
        else:
            candidates = [{"movie_id": item["movieId"], "retrieval_score": item["score"]}
                          for item in get_cold_start_recommendations(top_k=retrieval_candidates_k)]
                          
        # 2. Stage 2: Precision Ranking
        user_f = user_feats_map.get(uid)
        ranked = rank_candidates(
            candidates=candidates,
            user_features=user_f,
            item_features_lookup=item_feats_map,
            model_variant=model_name,
            top_k=top_k
        )
        
        lat_ms = (time.perf_counter() - t0) * 1000.0
        
        recommended_ids = [str(r["movie_id"]) for r in ranked]
        ground_truth = test_ground_truth.get(uid, {})
        relevant_ids = set(ground_truth.keys())
        
        # Evaluate metrics
        ndcg = compute_ndcg_at_k(recommended_ids, ground_truth, k=top_k)
        recall = compute_recall_at_k(recommended_ids, relevant_ids, k=top_k)
        hr = compute_hit_rate_at_k(recommended_ids, relevant_ids, k=top_k)
        prec = compute_precision_at_k(recommended_ids, relevant_ids, k=top_k)
        
        # Simulate clicks
        clicks = sum(simulate_user_click(mid, ground_truth) for mid in recommended_ids)
        
        # Record
        rec = metrics[variant]
        rec["users"] += 1
        rec["ndcg"].append(ndcg)
        rec["recall"].append(recall)
        rec["hit_rate"].append(hr)
        rec["precision"].append(prec)
        rec["clicks"] += clicks
        rec["impressions"] += len(recommended_ids)
        rec["latencies_ms"].append(lat_ms)
        
    # Statistical analysis
    a = metrics["Variant_A_DCN"]
    b = metrics["Variant_B_WideDeep"]
    
    a_ctr = compute_ctr(a["clicks"], a["impressions"])
    b_ctr = compute_ctr(b["clicks"], b["impressions"])
    
    t_stat_ndcg, p_val_ndcg = welch_t_test(a["ndcg"], b["ndcg"])
    t_stat_rec, p_val_rec = welch_t_test(a["recall"], b["recall"])
    
    ndcg_lift = ((np.mean(a["ndcg"]) - np.mean(b["ndcg"])) / (np.mean(b["ndcg"]) + 1e-9)) * 100.0
    ctr_lift = ((a_ctr - b_ctr) / (b_ctr + 1e-9)) * 100.0
    
    summary = {
        "total_users": len(eval_users),
        "variant_a_dcn": {
            "traffic_share": f"{a['users'] / len(eval_users):.1%}",
            "ndcg_at_10": round(float(np.mean(a["ndcg"])), 4),
            "recall_at_10": round(float(np.mean(a["recall"])), 4),
            "hit_rate_at_10": round(float(np.mean(a["hit_rate"])), 4),
            "ctr": round(a_ctr, 4),
            "latency_p50_ms": round(float(np.percentile(a["latencies_ms"], 50)), 2),
            "latency_p95_ms": round(float(np.percentile(a["latencies_ms"], 95)), 2)
        },
        "variant_b_wide_deep": {
            "traffic_share": f"{b['users'] / len(eval_users):.1%}",
            "ndcg_at_10": round(float(np.mean(b["ndcg"])), 4),
            "recall_at_10": round(float(np.mean(b["recall"])), 4),
            "hit_rate_at_10": round(float(np.mean(b["hit_rate"])), 4),
            "ctr": round(b_ctr, 4),
            "latency_p50_ms": round(float(np.percentile(b["latencies_ms"], 50)), 2),
            "latency_p95_ms": round(float(np.percentile(b["latencies_ms"], 95)), 2)
        },
        "statistical_tests": {
            "ndcg_lift": f"{ndcg_lift:+.2f}%",
            "ctr_lift": f"{ctr_lift:+.2f}%",
            "ndcg_p_value": round(p_val_ndcg, 4),
            "statistically_significant": bool(p_val_ndcg < 0.05),
            "winner": "Variant A (DCN-v2)" if ndcg_lift > 0 and p_val_ndcg < 0.05 else "Variant B or Inconclusive"
        }
    }
    
    print("\n[A/B Test Results Summary]:")
    print(f"  Total Requests: {len(eval_users):,}")
    print(f"  Variant A (DCN-v2)   -> NDCG@10: {summary['variant_a_dcn']['ndcg_at_10']} | Recall@10: {summary['variant_a_dcn']['recall_at_10']} | CTR: {summary['variant_a_dcn']['ctr']:.2%}")
    print(f"  Variant B (Wide&Deep)-> NDCG@10: {summary['variant_b_wide_deep']['ndcg_at_10']} | Recall@10: {summary['variant_b_wide_deep']['recall_at_10']} | CTR: {summary['variant_b_wide_deep']['ctr']:.2%}")
    print(f"  LIFT: NDCG {summary['statistical_tests']['ndcg_lift']} | CTR {summary['statistical_tests']['ctr_lift']} (p-value: {summary['statistical_tests']['ndcg_p_value']})")
    print(f"  Decision: {summary['statistical_tests']['winner']}\n")
    
    return {"summary": summary, "metrics": metrics}


if __name__ == "__main__":
    run_ab_experiment(num_test_users=200)
