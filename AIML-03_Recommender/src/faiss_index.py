"""
FAISS Vector Index for Fast Approximate Nearest-Neighbor (ANN) Candidate Retrieval.
Step 4 of AIML-03: Index item embeddings in FAISS for fast nearest-neighbor search.
Optimized for low RAM consumption with index persistence and item-to-item retrieval.
"""

import os
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class CandidateFAISSIndex:
    """
    Production FAISS Vector Index for Two-Stage Recommender Systems.
    Handles exact (IndexFlatIP) and approximate (IndexIVFFlat) candidate retrieval.
    """
    def __init__(self, dimension: int = 32, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.movie_ids = None
        self.movie_id_to_idx = {}
        
    def build(self, movie_embeddings: np.ndarray, movie_ids: np.ndarray, nlist: int = 64):
        """Build and populate FAISS index with L2-normalized item embeddings."""
        embeddings_copy = np.ascontiguousarray(movie_embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_copy)
        
        n_items = len(embeddings_copy)
        if self.index_type == "IVF" and n_items >= nlist * 10:
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings_copy)
            self.index.nprobe = 8
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            
        self.index.add(embeddings_copy)
        self.movie_ids = np.array([str(m) for m in movie_ids])
        self.movie_id_to_idx = {str(mid): i for i, mid in enumerate(self.movie_ids)}
        print(f"[FAISS] Built {self.index_type} index with {self.index.ntotal:,} items (dim={self.dimension}).")

    def search(self, user_embedding: np.ndarray, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves top_k candidate movie IDs with their cosine similarity scores.
        Returns: list of dicts: [{'movie_id': '10', 'retrieval_score': 0.84}, ...]
        """
        if self.index is None or self.movie_ids is None:
            raise RuntimeError("FAISS index has not been built or loaded.")
            
        query = np.ascontiguousarray(user_embedding.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)
        
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.movie_ids):
                candidates.append({
                    "movie_id": str(self.movie_ids[idx]),
                    "retrieval_score": float(score)
                })
        return candidates

    def get_similar_items(self, movie_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Item-to-item nearest-neighbor search for cold-start fallback.
        """
        if movie_id not in self.movie_id_to_idx:
            return []
            
        idx = self.movie_id_to_idx[movie_id]
        # Reconstruct vector or search from index
        item_vec = self.index.reconstruct(int(idx)).reshape(1, -1)
        scores, indices = self.index.search(item_vec, top_k + 1)
        
        similar_items = []
        for score, i in zip(scores[0], indices[0]):
            if i != -1 and i < len(self.movie_ids) and str(self.movie_ids[i]) != movie_id:
                similar_items.append({
                    "movie_id": str(self.movie_ids[i]),
                    "similarity": float(score)
                })
        return similar_items[:top_k]

    def save(self, file_path: str):
        """Save FAISS index and movie ID mapping to disk."""
        faiss.write_index(self.index, file_path)
        mapping_path = file_path + ".ids.npy"
        np.save(mapping_path, self.movie_ids)
        print(f"[FAISS] Saved index to {file_path}")

    def load(self, file_path: str):
        """Load FAISS index and movie ID mapping from disk with memory-mapping."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FAISS index file not found: {file_path}")
            
        self.index = faiss.read_index(file_path, faiss.IO_FLAG_MMAP)
        mapping_path = file_path + ".ids.npy"
        if os.path.exists(mapping_path):
            self.movie_ids = np.load(mapping_path, allow_pickle=True)
            self.movie_id_to_idx = {str(mid): i for i, mid in enumerate(self.movie_ids)}
        print(f"[FAISS] Loaded index with {self.index.ntotal:,} items from {file_path}")


def get_default_faiss_index() -> CandidateFAISSIndex:
    """Singleton-style factory to load or build default FAISS index."""
    index_path = os.path.join(MODELS_DIR, "faiss_index.bin")
    faiss_obj = CandidateFAISSIndex(dimension=32)
    
    if os.path.exists(index_path) and os.path.exists(index_path + ".ids.npy"):
        faiss_obj.load(index_path)
    else:
        movie_emb_path = os.path.join(MODELS_DIR, "movie_embeddings.npy")
        movie_ids_path = os.path.join(MODELS_DIR, "unique_movies.npy")
        if os.path.exists(movie_emb_path) and os.path.exists(movie_ids_path):
            movie_embs = np.load(movie_emb_path)
            movie_ids = np.load(movie_ids_path, allow_pickle=True)
            faiss_obj.build(movie_embs, movie_ids)
            try:
                faiss_obj.save(index_path)
            except Exception as e:
                print(f"[FAISS] Notice: could not persist index: {e}")
        else:
            raise RuntimeError("Embeddings not found to build FAISS index.")
            
    return faiss_obj


if __name__ == "__main__":
    idx = get_default_faiss_index()
    test_vec = np.random.randn(32).astype(np.float32)
    cands = idx.search(test_vec, top_k=5)
    print("Sample Candidate Retrieval:", cands)
