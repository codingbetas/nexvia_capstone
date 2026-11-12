"""
Evaluation Metrics for Recommendation Systems.
Step 7 of AIML-03: Measure NDCG, Recall@K, Hit Rate@K, Precision@K, and Click-Through Rate (CTR).
"""

import numpy as np
from typing import List, Set, Dict, Any, Union, Tuple


def compute_dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    relevance_scores = np.asarray(relevance_scores)[:k]
    if len(relevance_scores) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return float(np.sum((2.0 ** relevance_scores - 1.0) / discounts))


def compute_ndcg_at_k(recommended_ids: List[str], ground_truth_relevance: Dict[str, float], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K).
    ground_truth_relevance: mapping from movie_id to graded relevance (e.g. rating >= 4.0 -> 1.0).
    """
    if not ground_truth_relevance:
        return 0.0
        
    actual_rels = [ground_truth_relevance.get(str(mid), 0.0) for mid in recommended_ids[:k]]
    actual_dcg = compute_dcg_at_k(actual_rels, k)
    
    ideal_rels = sorted(list(ground_truth_relevance.values()), reverse=True)[:k]
    ideal_dcg = compute_dcg_at_k(ideal_rels, k)
    
    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def compute_recall_at_k(recommended_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Computes Recall@K: fraction of relevant items retrieved in top K."""
    if not relevant_ids:
        return 0.0
    top_k_rec = set(str(mid) for mid in recommended_ids[:k])
    hits = len(top_k_rec.intersection(relevant_ids))
    return float(hits / len(relevant_ids))


def compute_hit_rate_at_k(recommended_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Computes Hit Rate@K (1 if at least one relevant item is recommended, 0 otherwise)."""
    if not relevant_ids:
        return 0.0
    top_k_rec = set(str(mid) for mid in recommended_ids[:k])
    return 1.0 if len(top_k_rec.intersection(relevant_ids)) > 0 else 0.0


def compute_precision_at_k(recommended_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Computes Precision@K: fraction of top K items that are relevant."""
    if k <= 0:
        return 0.0
    top_k_rec = set(str(mid) for mid in recommended_ids[:k])
    hits = len(top_k_rec.intersection(relevant_ids))
    return float(hits / k)


def compute_ctr(total_clicks: int, total_impressions: int) -> float:
    """Computes Click-Through Rate."""
    if total_impressions <= 0:
        return 0.0
    return float(total_clicks / total_impressions)


def welch_t_test(sample_a: List[float], sample_b: List[float]) -> Tuple[float, float]:
    """
    Computes Welch's Two-Sample t-test for unequal variances.
    Returns: (t_statistic, p_value)
    """
    from scipy import stats
    res = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    return float(res.statistic), float(res.pvalue)
