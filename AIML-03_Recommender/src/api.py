"""
High-Performance Recommendation REST API (FastAPI).
Combines Stage 1 FAISS Candidate Retrieval, Stage 2 DCN/Wide&Deep Ranking,
Cold-Start Handling, Multi-Tier Caching, and Online A/B Testing.
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.faiss_index import get_default_faiss_index, CandidateFAISSIndex
from src.models.ranking_model import rank_candidates
from src.caching.redis_cache import RecommendationCache
from src.cold_start import get_cold_start_recommendations, COLD_START_HANDLER
from src.triton_client import TRITON_BRIDGE
from src.data.dataset import get_movie_metadata, CURATED_MOVIES

app = FastAPI(
    title="Large-Scale Personalized Recommendation Engine",
    description="Two-Stage Recommender (FAISS Candidate Retrieval + DCN-v2 / Wide & Deep Precision Ranking)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# 1. Load Precomputed Embeddings and Fast Lookups (O(1) Hashmaps)
print("[API] Loading embeddings and candidate index...")
USER_EMB = np.load(os.path.join(MODELS_DIR, "user_embeddings.npy"))
UNIQUE_USERS = np.load(os.path.join(MODELS_DIR, "unique_users.npy"), allow_pickle=True)
UNIQUE_MOVIES = np.load(os.path.join(MODELS_DIR, "unique_movies.npy"), allow_pickle=True)

USER_TO_IDX = {str(uid): i for i, uid in enumerate(UNIQUE_USERS)}
MOVIE_TO_IDX = {str(mid): i for i, mid in enumerate(UNIQUE_MOVIES)}

# FAISS Vector Index
faiss_index = get_default_faiss_index()

# Multi-Tier Resilient Cache (L1 LRU + L2 Redis)
cache = RecommendationCache()

# Load Item and User Feature Maps for Stage 2 Ranker
user_feats_map = {}
item_feats_map = {}
user_feat_path = os.path.join(DATA_DIR, "user_features.parquet")
item_feat_path = os.path.join(DATA_DIR, "item_features.parquet")

if os.path.exists(user_feat_path):
    u_df = pd.read_parquet(user_feat_path)
    user_feats_map = {str(row["userId"]): row.to_dict() for _, row in u_df.iterrows()}

if os.path.exists(item_feat_path):
    i_df = pd.read_parquet(item_feat_path)
    item_feats_map = {str(row["movieId"]): row.to_dict() for _, row in i_df.iterrows()}

TELEMETRY = {
    "requests_total": 0,
    "Variant_A_DCN": {"requests": 0, "impressions": 0, "clicks": 0, "latencies_ms": []},
    "Variant_B_WideDeep": {"requests": 0, "impressions": 0, "clicks": 0, "latencies_ms": []}
}


def get_ab_variant(user_id: str) -> str:
    """Deterministic A/B Traffic Split via MD5 hash modulo 2."""
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "Variant_A_DCN" if hash_val % 2 == 0 else "Variant_B_WideDeep"


# --- Request and Response Schemas ---
class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="Unique User Identifier")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    preferred_genres: Optional[List[str]] = Field(None, description="Preferred genres for cold-start exploration")
    model_variant: Optional[str] = Field(None, description="Force variant: Variant_A_DCN or Variant_B_WideDeep")


class MovieItem(BaseModel):
    movie_id: str
    title: str
    genres: str
    year: int
    retrieval_score: float
    ranking_score: float
    stage1_rank: Optional[int] = None
    stage2_rank: Optional[int] = None
    rank_delta: Optional[int] = None
    rerank_reason: Optional[str] = None


class RecommendResponse(BaseModel):
    user_id: str
    variant: str
    is_cold_start: bool
    cache_hit: bool
    inference_source: str
    latency_ms: float
    stage1_candidates: List[MovieItem] = []
    recommendations: List[MovieItem]


class ClickFeedback(BaseModel):
    user_id: str
    movie_id: str
    variant: str


# --- Endpoints ---
@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(req: RecommendRequest):
    """
    Two-Stage Recommendation Pipeline:
    1. Check L1/L2 Cache
    2. Route traffic to A/B variant (DCN-v2 vs Wide & Deep)
    3. Cold-Start Fallback (Popularity or Content-based) if user has no interactions
    4. Stage 1: FAISS Vector Retrieval (Top-50 Candidates)
    5. Stage 2: Precision Ranking Model (Re-score to Top-K with Rank Movement explanation)
    6. Return both Stage 1 Candidates & Stage 2 Ranked Recommendations
    """
    t0 = time.perf_counter()
    TELEMETRY["requests_total"] += 1
    
    variant = req.model_variant or get_ab_variant(req.user_id)
    model_name = "DCN-v2" if "DCN" in variant else "Wide&Deep"
    
    # 1. Check Cache
    cached_payload = cache.get_cached_recommendations(req.user_id)
    if cached_payload and not req.preferred_genres:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        TELEMETRY[variant]["requests"] += 1
        TELEMETRY[variant]["impressions"] += len(cached_payload.get("recommendations", [])[:req.top_k])
        TELEMETRY[variant]["latencies_ms"].append(lat_ms)
        return RecommendResponse(
            user_id=req.user_id,
            variant=variant,
            is_cold_start=False,
            cache_hit=True,
            inference_source="Redis/L1-Cache",
            latency_ms=round(lat_ms, 2),
            stage1_candidates=cached_payload.get("stage1_candidates", [])[:req.top_k],
            recommendations=cached_payload.get("recommendations", [])[:req.top_k]
        )

    # 2. Cold-Start Handling
    is_cold_start = req.user_id not in USER_TO_IDX
    if is_cold_start:
        raw_items = get_cold_start_recommendations(
            preferred_genres=req.preferred_genres,
            top_k=req.top_k
        )
        recommendations = []
        for rank_idx, item in enumerate(raw_items, 1):
            mid = str(item.get("movieId", item.get("movie_id")))
            meta = get_movie_metadata(mid)
            recommendations.append(MovieItem(
                movie_id=mid,
                title=meta["title"],
                genres=meta["genres"],
                year=meta["year"],
                retrieval_score=round(float(item.get("score", 0.8)), 3),
                ranking_score=round(float(item.get("score", 0.8)), 3),
                stage1_rank=rank_idx,
                stage2_rank=rank_idx,
                rank_delta=0,
                rerank_reason="Cold-Start Fallback: Bayesian Wilson Popularity & Genre Match"
            ))
            
        lat_ms = (time.perf_counter() - t0) * 1000.0
        TELEMETRY[variant]["requests"] += 1
        TELEMETRY[variant]["impressions"] += len(recommendations)
        TELEMETRY[variant]["latencies_ms"].append(lat_ms)
        
        return RecommendResponse(
            user_id=req.user_id,
            variant=variant,
            is_cold_start=True,
            cache_hit=False,
            inference_source="ColdStart-Fallback",
            latency_ms=round(lat_ms, 2),
            stage1_candidates=recommendations,
            recommendations=recommendations
        )

    # 3. Stage 1: Retrieval via FAISS (Retrieve 50 Candidates)
    u_idx = USER_TO_IDX[req.user_id]
    user_vector = USER_EMB[u_idx]
    user_vector, inf_source = TRITON_BRIDGE.infer_user_embedding(req.user_id, user_vector)
    
    candidate_pool_k = max(req.top_k * 4, 40)
    raw_candidates = faiss_index.search(user_vector, top_k=candidate_pool_k)

    # Track Stage 1 initial ranking
    for s1_pos, cand in enumerate(raw_candidates, 1):
        cand["stage1_rank"] = s1_pos

    # Build Stage 1 Candidate Items list (Top 12 for side-by-side display)
    stage1_items = []
    for cand in raw_candidates[:max(req.top_k, 10)]:
        mid = str(cand["movie_id"])
        meta = get_movie_metadata(mid)
        stage1_items.append(MovieItem(
            movie_id=mid,
            title=meta["title"],
            genres=meta["genres"],
            year=meta["year"],
            retrieval_score=round(float(cand.get("retrieval_score", 0.0)), 4),
            ranking_score=round(float(cand.get("retrieval_score", 0.0)), 4),
            stage1_rank=cand["stage1_rank"],
            stage2_rank=None,
            rank_delta=0,
            rerank_reason="Retrieved via 32-dim Cosine Similarity in FAISS Space"
        ))

    # 4. Stage 2: Precision Ranking Model (DCN-v2 / Wide & Deep)
    user_f = user_feats_map.get(req.user_id)
    ranked_candidates = rank_candidates(
        candidates=raw_candidates,
        user_features=user_f,
        item_features_lookup=item_feats_map,
        model_variant=model_name,
        top_k=req.top_k
    )

    # 5. Hydrate Stage 2 Final Recommendations with Rank Movement Deltas
    recommendations = []
    for s2_pos, cand in enumerate(ranked_candidates, 1):
        mid = str(cand["movie_id"])
        meta = get_movie_metadata(mid)
        s1_rank = cand.get("stage1_rank", s2_pos)
        delta = s1_rank - s2_pos  # Positive: promoted up, Negative: dropped down
        
        if delta > 0:
            reason = f"[+ {delta} Promoted] Stage 1 #{s1_rank} -> Stage 2 #{s2_pos} via {model_name}"
        elif delta < 0:
            reason = f"[- {abs(delta)} Demoted] Stage 1 #{s1_rank} -> Stage 2 #{s2_pos} via {model_name}"
        else:
            reason = f"[Kept #{s2_pos}] Maintained rank across both stages"

        recommendations.append(MovieItem(
            movie_id=mid,
            title=meta["title"],
            genres=meta["genres"],
            year=meta["year"],
            retrieval_score=round(float(cand.get("retrieval_score", 0.0)), 4),
            ranking_score=round(float(cand.get("ranking_score", 0.0)), 4),
            stage1_rank=s1_rank,
            stage2_rank=s2_pos,
            rank_delta=delta,
            rerank_reason=reason
        ))

    # 6. Cache both stages for instant future retrieval
    cache_data = {
        "stage1_candidates": [c.dict() for c in stage1_items],
        "recommendations": [r.dict() for r in recommendations]
    }
    cache.cache_recommendations(req.user_id, cache_data)
    
    lat_ms = (time.perf_counter() - t0) * 1000.0
    TELEMETRY[variant]["requests"] += 1
    TELEMETRY[variant]["impressions"] += len(recommendations)
    TELEMETRY[variant]["latencies_ms"].append(lat_ms)

    return RecommendResponse(
        user_id=req.user_id,
        variant=variant,
        is_cold_start=False,
        cache_hit=False,
        inference_source=inf_source,
        latency_ms=round(lat_ms, 2),
        stage1_candidates=stage1_items,
        recommendations=recommendations
    )


@app.post("/feedback/click")
def record_click(feedback: ClickFeedback):
    """Logs online interaction feedback to compute real-time Click-Through Rate (CTR)."""
    if feedback.variant in TELEMETRY:
        TELEMETRY[feedback.variant]["clicks"] += 1
        return {"status": "success", "variant": feedback.variant, "total_clicks": TELEMETRY[feedback.variant]["clicks"]}
    return {"status": "error", "message": "Unknown variant"}


@app.get("/metrics")
def get_metrics():
    """Returns real-time telemetry, cache hit ratios, and online A/B test statistics."""
    cache_stats = cache.get_stats()
    
    a = TELEMETRY["Variant_A_DCN"]
    b = TELEMETRY["Variant_B_WideDeep"]
    
    a_ctr = (a["clicks"] / a["impressions"]) if a["impressions"] > 0 else 0.0
    b_ctr = (b["clicks"] / b["impressions"]) if b["impressions"] > 0 else 0.0
    
    a_lat = np.mean(a["latencies_ms"]) if a["latencies_ms"] else 0.0
    b_lat = np.mean(b["latencies_ms"]) if b["latencies_ms"] else 0.0
    
    return {
        "total_requests": TELEMETRY["requests_total"],
        "cache": cache_stats,
        "ab_experiment": {
            "variant_a_dcn": {
                "requests": a["requests"],
                "impressions": a["impressions"],
                "clicks": a["clicks"],
                "ctr": round(a_ctr, 4),
                "avg_latency_ms": round(float(a_lat), 2)
            },
            "variant_b_wide_deep": {
                "requests": b["requests"],
                "impressions": b["impressions"],
                "clicks": b["clicks"],
                "ctr": round(b_ctr, 4),
                "avg_latency_ms": round(float(b_lat), 2)
            },
            "ctr_lift": f"{((a_ctr - b_ctr) / (b_ctr + 1e-9) * 100):+.2f}%" if b_ctr > 0 else "N/A"
        },
        "triton_status": TRITON_BRIDGE.get_status()
    }


@app.get("/movies/search")
def search_movies(query: str = Query(..., min_length=1), limit: int = 10):
    """Search movies by title or genre."""
    q = query.lower()
    matches = []
    
    for mid, m in CURATED_MOVIES.items():
        if q in m["title"].lower() or q in m["genres"].lower():
            matches.append({"movieId": mid, "title": m["title"], "genres": m["genres"], "year": m["year"]})
            if len(matches) >= limit:
                return matches
                
    return matches


@app.get("/health")
def health():
    """System health check and memory usage diagnostic."""
    return {
        "status": "healthy",
        "faiss_total_movies": faiss_index.index.ntotal if faiss_index.index else 0,
        "unique_users_loaded": len(UNIQUE_USERS),
        "unique_movies_loaded": len(UNIQUE_MOVIES),
        "cache_l1_items": len(cache.l1_cache.cache),
        "redis_connected": cache.redis_client is not None,
        "triton_connected": TRITON_BRIDGE.is_connected
    }
