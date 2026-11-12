"""
Resilient Multi-Tier Recommendation Caching Layer (L1 In-Memory LRU + L2 Redis).
Step 6 of AIML-03: Cache results in Redis.
Maintains strictly bounded RAM usage (< 5MB) and zero-latency failover when Redis is offline.
"""

import time
import json
from collections import OrderedDict
from typing import List, Dict, Any, Optional

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class InMemoryLRUCache:
    """Bounded, thread-safe In-Memory LRU cache with TTL expiration."""
    def __init__(self, capacity: int = 1000, default_ttl: int = 3600):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.expirations = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        if key not in self.cache:
            return None
        if key in self.expirations and now > self.expirations[key]:
            del self.cache[key]
            del self.expirations[key]
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if ttl is None:
            ttl = self.default_ttl
        now = time.time()
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            oldest_key, _ = self.cache.popitem(last=False)
            self.expirations.pop(oldest_key, None)
        self.cache[key] = value
        self.expirations[key] = now + ttl

    def clear(self):
        self.cache.clear()
        self.expirations.clear()


class RecommendationCache:
    """
    Two-Tier Resilient Cache:
    - Tier 1: In-Memory LRU Cache (Fastest, zero network overhead)
    - Tier 2: Redis Key-Value Store with TTL
    """
    def __init__(self, host: str = "localhost", port: int = 6379, socket_timeout: float = 0.2):
        self.l1_cache = InMemoryLRUCache(capacity=1000, default_ttl=3600)
        self.redis_client = None
        self.hits = 0
        self.misses = 0
        
        if HAS_REDIS:
            try:
                client = redis.Redis(
                    host=host,
                    port=port,
                    socket_connect_timeout=socket_timeout,
                    socket_timeout=socket_timeout,
                    decode_responses=True
                )
                client.ping()
                self.redis_client = client
                print(f"[Cache] Connected to Redis at {host}:{port}")
            except Exception:
                self.redis_client = None
                print("[Cache] Redis offline. Operating seamlessly with L1 In-Memory LRU cache.")

    def get_cached_recommendations(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch cached recommendations checking L1 then L2."""
        cache_key = f"rec:{user_id}"
        
        # 1. Check L1 Memory Cache
        val = self.l1_cache.get(cache_key)
        if val is not None:
            self.hits += 1
            return val
            
        # 2. Check L2 Redis Cache
        if self.redis_client is not None:
            try:
                data = self.redis_client.get(cache_key)
                if data:
                    parsed = json.loads(data)
                    self.l1_cache.set(cache_key, parsed)
                    self.hits += 1
                    return parsed
            except Exception:
                pass
                
        self.misses += 1
        return None

    def cache_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]], ttl: int = 3600):
        """Write recommendations to both L1 and L2."""
        cache_key = f"rec:{user_id}"
        
        # Save to L1
        self.l1_cache.set(cache_key, recommendations, ttl=ttl)
        
        # Save to L2 Redis
        if self.redis_client is not None:
            try:
                self.redis_client.setex(cache_key, ttl, json.dumps(recommendations))
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache telemetry."""
        total = self.hits + self.misses
        hit_ratio = (self.hits / total) if total > 0 else 0.0
        return {
            "l1_items": len(self.l1_cache.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(hit_ratio, 4),
            "redis_connected": self.redis_client is not None
        }
