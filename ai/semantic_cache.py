"""
DevMentor AI — Semantic Cache
==============================
Production-grade semantic caching with:

- Vector similarity search via Qdrant for cache lookup
- Redis for fast exact-match and metadata storage
- Embedding generation with model abstraction
- Configurable similarity threshold
- TTL-based cache expiration
- Cache statistics and hit rate tracking
- Async-ready design
- Graceful degradation on failures
- Cache warming and invalidation utilities

How it works:
    1. User sends a query
    2. Generate embedding for the query
    3. Search Qdrant for semantically similar cached queries
    4. If similarity >= threshold → return cached response (cache HIT)
    5. If no match → call LLM → store result in cache (cache MISS)

This saves significant API costs when users ask similar questions.

Usage:
    from ai.semantic_cache import SemanticCache

    cache = SemanticCache()

    # Check cache before LLM call
    cached = cache.get(user_query)
    if cached:
        return cached

    # Call LLM
    response = call_llm(user_query)

    # Store in cache
    cache.set(user_query, response)
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import get_settings

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_COLLECTION = "semantic_query_cache"
EXACT_CACHE_PREFIX = "scache:exact:"
STATS_KEY = "scache:stats"
METADATA_PREFIX = "scache:meta:"


# ===========================================================================
# Data Classes
# ===========================================================================

@dataclass
class CacheEntry:
    """A single semantic cache entry."""
    query: str
    response: str
    model: str
    tokens_used: int
    created_at: str
    hit_count: int = 0
    last_accessed: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheResult:
    """Result of a cache lookup."""
    hit: bool
    response: Optional[str] = None
    similarity_score: float = 0.0
    hit_type: str = "none"        # exact | semantic | none
    cached_query: Optional[str] = None
    entry_id: Optional[str] = None


# ===========================================================================
# Embedding Generator
# ===========================================================================

class EmbeddingGenerator:
    """
    Abstraction layer for embedding generation.
    Supports multiple embedding models with fallback.
    """

    def __init__(self):
        self._model = None
        self._model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._vector_size = settings.qdrant_vector_size

    def _load_model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded: %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            self._model = None
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            self._model = None

    def generate(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            numpy array of shape (vector_size,) or None on failure
        """
        self._load_model()

        if self._model is None:
            return self._fallback_embedding(text)

        try:
            # Truncate very long texts
            text = text[:2048] if len(text) > 2048 else text
            embedding = self._model.encode(text, normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Deterministic fallback embedding using character hashing.
        Not semantically meaningful but allows cache to function
        in degraded mode (only exact matches will work).
        """
        logger.warning("Using fallback embedding — semantic matching degraded")
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._vector_size).astype(np.float32)
        # Normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def batch_generate(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts efficiently."""
        self._load_model()

        if self._model is None:
            return [self._fallback_embedding(t) for t in texts]

        try:
            texts = [t[:2048] for t in texts]
            embeddings = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as exc:
            logger.error("Batch embedding failed: %s", exc)
            return [self._fallback_embedding(t) for t in texts]


# ===========================================================================
# Redis Client Helper
# ===========================================================================

def _get_redis():
    """Get Redis client, returns None if unavailable."""
    try:
        from redis_client import get_redis_client
        client = get_redis_client()
        client.ping()
        return client
    except Exception as exc:
        logger.warning("Redis unavailable for semantic cache: %s", exc)
        return None


# ===========================================================================
# Qdrant Client Helper
# ===========================================================================

def _get_qdrant():
    """Get Qdrant client, returns None if unavailable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=settings.qdrant_timeout,
        )
        return client
    except Exception as exc:
        logger.warning("Qdrant unavailable for semantic cache: %s", exc)
        return None


# ===========================================================================
# Semantic Cache
# ===========================================================================

class SemanticCache:
    """
    Production-grade semantic cache combining:
    - Qdrant: vector similarity search for semantic matching
    - Redis: exact match cache + metadata + statistics

    Two-level lookup:
    1. Exact match (Redis) — O(1), fastest
    2. Semantic match (Qdrant) — finds similar questions
    """

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        collection_name: str = CACHE_COLLECTION,
    ):
        self.threshold = similarity_threshold or settings.semantic_cache_similarity_threshold
        self.ttl = ttl_seconds or settings.semantic_cache_ttl_seconds
        self.collection = collection_name
        self.embedder = EmbeddingGenerator()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        qdrant = _get_qdrant()
        if not qdrant:
            return

        try:
            from qdrant_client.models import Distance, VectorParams
            collections = [c.name for c in qdrant.get_collections().collections]

            if self.collection not in collections:
                qdrant.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=settings.qdrant_vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created semantic cache collection: %s", self.collection)
        except Exception as exc:
            logger.error("Failed to ensure cache collection: %s", exc)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get(
        self,
        query: str,
        namespace: Optional[str] = None,
    ) -> CacheResult:
        """
        Look up a query in the semantic cache.

        Args:
            query:     User's query text
            namespace: Optional namespace to scope cache (e.g. user_id, mode)

        Returns:
            CacheResult with hit=True and response if found
        """
        if not settings.semantic_cache_enabled:
            return CacheResult(hit=False)

        if not query or not query.strip():
            return CacheResult(hit=False)

        query = query.strip()

        # ---- Level 1: Exact match via Redis ----
        exact_result = self._exact_match(query, namespace)
        if exact_result.hit:
            self._increment_stats("hits")
            self._increment_stats("exact_hits")
            logger.debug("Semantic cache EXACT HIT for query: %s...", query[:50])
            return exact_result

        # ---- Level 2: Semantic match via Qdrant ----
        semantic_result = self._semantic_match(query, namespace)
        if semantic_result.hit:
            self._increment_stats("hits")
            self._increment_stats("semantic_hits")
            logger.info(
                "Semantic cache SEMANTIC HIT (score=%.3f): %s...",
                semantic_result.similarity_score,
                query[:50],
            )
            return semantic_result

        self._increment_stats("misses")
        logger.debug("Semantic cache MISS for query: %s...", query[:50])
        return CacheResult(hit=False)

    def set(
        self,
        query: str,
        response: str,
        model: str = "unknown",
        tokens_used: int = 0,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store a query-response pair in the semantic cache.

        Args:
            query:      User's query text
            response:   LLM response to cache
            model:      Model that generated the response
            tokens_used: Token count for cost tracking
            namespace:  Optional namespace scope
            metadata:   Additional metadata to store

        Returns:
            True if successfully cached
        """
        if not settings.semantic_cache_enabled:
            return False

        if not query or not response:
            return False

        query = query.strip()
        entry_id = self._make_id(query, namespace)

        entry = CacheEntry(
            query=query,
            response=response,
            model=model,
            tokens_used=tokens_used,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        # Store in Redis (exact match + metadata)
        redis_success = self._store_exact(query, entry, entry_id, namespace)

        # Store in Qdrant (semantic search)
        qdrant_success = self._store_semantic(query, entry, entry_id, namespace)

        if redis_success or qdrant_success:
            self._increment_stats("stored")
            logger.debug("Cached response for query: %s...", query[:50])
            return True

        return False

    def invalidate(self, query: str, namespace: Optional[str] = None) -> bool:
        """
        Remove a specific query from the cache.

        Args:
            query:     Query to invalidate
            namespace: Optional namespace scope

        Returns:
            True if removed
        """
        query = query.strip()
        entry_id = self._make_id(query, namespace)

        redis = _get_redis()
        qdrant = _get_qdrant()

        success = False

        if redis:
            try:
                exact_key = self._exact_key(query, namespace)
                meta_key = f"{METADATA_PREFIX}{entry_id}"
                redis.delete(exact_key, meta_key)
                success = True
            except Exception as exc:
                logger.error("Redis invalidation failed: %s", exc)

        if qdrant:
            try:
                from qdrant_client.models import PointIdsList
                qdrant.delete(
                    collection_name=self.collection,
                    points_selector=PointIdsList(points=[entry_id]),
                )
                success = True
            except Exception as exc:
                logger.error("Qdrant invalidation failed: %s", exc)

        if success:
            logger.info("Cache entry invalidated: %s...", query[:50])

        return success

    def flush(self, namespace: Optional[str] = None) -> bool:
        """
        Clear all cache entries (or all entries in a namespace).

        Args:
            namespace: If provided, only flush this namespace

        Returns:
            True if flushed successfully
        """
        redis = _get_redis()
        qdrant = _get_qdrant()

        if namespace:
            # Flush specific namespace
            if redis:
                try:
                    pattern = f"{EXACT_CACHE_PREFIX}{namespace}:*"
                    keys = redis.keys(pattern)
                    if keys:
                        redis.delete(*keys)
                except Exception as exc:
                    logger.error("Redis namespace flush failed: %s", exc)
        else:
            # Flush entire collection
            if qdrant:
                try:
                    qdrant.delete_collection(self.collection)
                    self._ensure_collection()
                except Exception as exc:
                    logger.error("Qdrant collection flush failed: %s", exc)

            if redis:
                try:
                    keys = redis.keys(f"{EXACT_CACHE_PREFIX}*")
                    meta_keys = redis.keys(f"{METADATA_PREFIX}*")
                    all_keys = list(keys) + list(meta_keys)
                    if all_keys:
                        redis.delete(*all_keys)
                except Exception as exc:
                    logger.error("Redis full flush failed: %s", exc)

        logger.info("Cache flushed (namespace=%s)", namespace or "all")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dict with hit rate, total hits, misses, stored count
        """
        redis = _get_redis()

        if not redis:
            return {"error": "Redis unavailable", "hit_rate": 0}

        try:
            raw = redis.hgetall(STATS_KEY)
            stats = {k.decode() if isinstance(k, bytes) else k:
                     int(v) for k, v in raw.items()}

            total = stats.get("hits", 0) + stats.get("misses", 0)
            hit_rate = (stats.get("hits", 0) / total * 100) if total > 0 else 0

            return {
                "hits": stats.get("hits", 0),
                "misses": stats.get("misses", 0),
                "exact_hits": stats.get("exact_hits", 0),
                "semantic_hits": stats.get("semantic_hits", 0),
                "stored": stats.get("stored", 0),
                "total_requests": total,
                "hit_rate_percent": round(hit_rate, 2),
                "threshold": self.threshold,
                "ttl_seconds": self.ttl,
            }
        except Exception as exc:
            logger.error("Failed to get cache stats: %s", exc)
            return {"error": str(exc)}

    # -----------------------------------------------------------------------
    # Private: Exact Match (Redis)
    # -----------------------------------------------------------------------

    def _exact_key(self, query: str, namespace: Optional[str]) -> str:
        """Build Redis key for exact match lookup."""
        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()
        if namespace:
            return f"{EXACT_CACHE_PREFIX}{namespace}:{query_hash}"
        return f"{EXACT_CACHE_PREFIX}{query_hash}"

    def _exact_match(self, query: str, namespace: Optional[str]) -> CacheResult:
        """Check Redis for exact query match."""
        redis = _get_redis()
        if not redis:
            return CacheResult(hit=False)

        try:
            key = self._exact_key(query, namespace)
            data = redis.get(key)

            if not data:
                return CacheResult(hit=False)

            entry_data = json.loads(data)
            response = entry_data.get("response")

            if not response:
                return CacheResult(hit=False)

            # Update hit count
            entry_data["hit_count"] = entry_data.get("hit_count", 0) + 1
            entry_data["last_accessed"] = datetime.now(timezone.utc).isoformat()
            redis.setex(key, self.ttl, json.dumps(entry_data))

            return CacheResult(
                hit=True,
                response=response,
                similarity_score=1.0,
                hit_type="exact",
                cached_query=entry_data.get("query"),
            )
        except Exception as exc:
            logger.error("Exact match lookup failed: %s", exc)
            return CacheResult(hit=False)

    def _store_exact(
        self,
        query: str,
        entry: CacheEntry,
        entry_id: str,
        namespace: Optional[str],
    ) -> bool:
        """Store entry in Redis for exact match lookup."""
        redis = _get_redis()
        if not redis:
            return False

        try:
            key = self._exact_key(query, namespace)
            data = {
                "query": entry.query,
                "response": entry.response,
                "model": entry.model,
                "tokens_used": entry.tokens_used,
                "created_at": entry.created_at,
                "hit_count": 0,
                "entry_id": entry_id,
            }
            redis.setex(key, self.ttl, json.dumps(data))
            return True
        except Exception as exc:
            logger.error("Exact cache store failed: %s", exc)
            return False

    # -----------------------------------------------------------------------
    # Private: Semantic Match (Qdrant)
    # -----------------------------------------------------------------------

    def _semantic_match(self, query: str, namespace: Optional[str]) -> CacheResult:
        """Search Qdrant for semantically similar cached queries."""
        qdrant = _get_qdrant()
        if not qdrant:
            return CacheResult(hit=False)

        embedding = self.embedder.generate(query)
        if embedding is None:
            return CacheResult(hit=False)

        try:
            search_filter = None
            if namespace:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[FieldCondition(
                        key="namespace",
                        match=MatchValue(value=namespace),
                    )]
                )

            hits = qdrant.search(
                collection_name=self.collection,
                query_vector=embedding.tolist(),
                limit=1,
                score_threshold=self.threshold,
                query_filter=search_filter,
                with_payload=True,
            )

            if not hits:
                return CacheResult(hit=False)

            best = hits[0]
            payload = best.payload or {}
            response = payload.get("response")

            if not response:
                return CacheResult(hit=False)

            return CacheResult(
                hit=True,
                response=response,
                similarity_score=best.score,
                hit_type="semantic",
                cached_query=payload.get("query"),
                entry_id=str(best.id),
            )
        except Exception as exc:
            logger.error("Semantic cache search failed: %s", exc)
            return CacheResult(hit=False)

    def _store_semantic(
        self,
        query: str,
        entry: CacheEntry,
        entry_id: str,
        namespace: Optional[str],
    ) -> bool:
        """Store entry in Qdrant for semantic search."""
        qdrant = _get_qdrant()
        if not qdrant:
            return False

        embedding = self.embedder.generate(query)
        if embedding is None:
            return False

        try:
            from qdrant_client.models import PointStruct

            payload = {
                "query": entry.query,
                "response": entry.response,
                "model": entry.model,
                "tokens_used": entry.tokens_used,
                "created_at": entry.created_at,
                "namespace": namespace or "default",
                **entry.metadata,
            }

            qdrant.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=entry_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception as exc:
            logger.error("Semantic cache store failed: %s", exc)
            return False

    # -----------------------------------------------------------------------
    # Private: Utilities
    # -----------------------------------------------------------------------

    def _make_id(self, query: str, namespace: Optional[str]) -> str:
        """Generate a deterministic UUID for a query."""
        key = f"{namespace or 'default'}:{query.lower().strip()}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return str(uuid.UUID(digest[:32]))

    def _increment_stats(self, stat: str) -> None:
        """Increment a stats counter in Redis."""
        redis = _get_redis()
        if redis:
            try:
                redis.hincrby(STATS_KEY, stat, 1)
            except Exception:
                pass


# ===========================================================================
# Module-level singleton
# ===========================================================================
_cache_instance: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the global SemanticCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance


# ===========================================================================
# Convenience functions (backward compatible)
# ===========================================================================

def semantic_cache_lookup(query: str, namespace: Optional[str] = None) -> Optional[str]:
    """
    Simple lookup — returns response string or None.
    Backward compatible with old semantic_cache_key() function.
    """
    result = get_semantic_cache().get(query, namespace)
    return result.response if result.hit else None


def cache_response(
    query: str,
    response: str,
    model: str = "unknown",
    tokens_used: int = 0,
    namespace: Optional[str] = None,
) -> bool:
    """
    Simple cache store. Backward compatible with old cache_response() function.
    """
    return get_semantic_cache().set(query, response, model, tokens_used, namespace)