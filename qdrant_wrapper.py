"""
DevMentor AI — Qdrant Vector Database Client Wrapper

RENAMED from qdrant_client.py to qdrant_wrapper.py — fixes a real,
project-wide bug found running the app for the first time.

THE BUG: this file used to be named qdrant_client.py, sitting at the
project root. The actual third-party vector DB library is ALSO a pip
package named `qdrant_client`. Because this local file lived at the
project root (on Python's import path) with the exact same name as the
installed package, every `from qdrant_client import QdrantClient`
anywhere in the codebase — including inside this file itself, in
get_qdrant_client() — resolved to THIS file instead of the real package.
Since this file has no class named QdrantClient, every one of those
imports failed at runtime with:
    ImportError: cannot import name 'QdrantClient' from 'qdrant_client'
This is exactly the warning long_term_memory.py logged on first boot.
The same break silently affected every other file that imported the real
qdrant_client package by that name — ai/semantic_cache.py, monitoring.py,
background_tasks.py, and this file's own internal use, since
get_qdrant_client() tried to import QdrantClient from "qdrant_client"
while itself BEING the module Python resolved that name to.

THE FIX: renamed this file so it no longer shares a name with the
installed package. Anything that needs THIS wrapper's functions
(get_qdrant_wrapper, AsyncQdrantWrapper, get_qdrant_client) must now
import from `qdrant_wrapper`, not `qdrant_client`. Anything that needs the
REAL third-party library continues to import `from qdrant_client import
QdrantClient` exactly as before — those imports now work correctly since
nothing local shadows that name anymore.

Production-grade vector DB client with:
- Async wrapper around sync Qdrant client (thread pool execution)
- Singleton embedding model with lazy loading
- Per-user collection support (isolation)
- Retry logic with exponential backoff
- Hybrid search support (vector + metadata filtering)
- Pagination-safe bulk delete
- Health checks for monitoring
- Graceful degradation on connection failure

Usage:
    from qdrant_wrapper import get_qdrant_wrapper

    qdrant = get_qdrant_wrapper()
    point_id = await qdrant.store_memory(user_id, text, metadata)
    results = await qdrant.search_memory(user_id, query, limit=5)
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

COLLECTION_NAME = settings.qdrant_collection_name
VECTOR_SIZE = settings.qdrant_vector_size


# ===========================================================================
# Embedding Model Singleton
# ===========================================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the sentence transformer model once."""
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded: %s", model_name)
        return model
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        return None


# ===========================================================================
# Qdrant Sync Client Singleton
# ===========================================================================

_qdrant_client = None


def get_qdrant_client():
    """Get or create the singleton Qdrant client, ensuring collection exists."""
    global _qdrant_client
    if _qdrant_client is None:
        # FIX: this now correctly resolves to the real third-party package,
        # since this file is no longer also named qdrant_client and
        # therefore no longer shadows it.
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=settings.qdrant_timeout,
        )
        _ensure_collection(_qdrant_client, COLLECTION_NAME)
    return _qdrant_client


def _ensure_collection(client, collection_name: str) -> None:
    """Create a Qdrant collection if it doesn't already exist."""
    try:
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", collection_name)
    except Exception as exc:
        logger.error("Failed to ensure collection %s: %s", collection_name, exc)


# ===========================================================================
# Async Qdrant Wrapper
# ===========================================================================

class AsyncQdrantWrapper:
    """
    Async-friendly wrapper around the sync Qdrant client.
    Blocking calls run in a thread pool via asyncio.to_thread.
    """

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self._client = None
        self._encoder = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = get_embedding_model()
        return self._encoder

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding, returns None if encoder unavailable."""
        if self.encoder is None:
            return None
        try:
            return self.encoder.encode(text[:2048], normalize_embeddings=True).tolist()
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def store_memory(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Store a text + embedding as a point in Qdrant.

        Args:
            user_id:  Owner of this memory
            text:     Text content to embed and store
            metadata: Additional payload fields

        Returns:
            Point ID if stored successfully, None on failure
        """
        def _sync_store():
            vector = self._embed(text)
            if vector is None:
                return None

            from qdrant_client.models import PointStruct
            point_id = str(uuid.uuid4())
            payload = {
                "user_id": user_id,
                "text": text,
                "metadata": metadata or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            return point_id

        return await asyncio.to_thread(_sync_store)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def search_memory(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        Search for semantically similar memories belonging to a user.

        Args:
            user_id:         Owner to filter by
            query:           Search query text
            limit:           Max results
            score_threshold: Minimum similarity score
            extra_filters:   Additional payload field filters {field: value}

        Returns:
            List of dicts with text, score, metadata, timestamp
        """
        def _sync_search():
            vector = self._embed(query)
            if vector is None:
                return []

            from qdrant_client.models import Filter, FieldCondition, MatchValue

            must_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
            if extra_filters:
                for field, value in extra_filters.items():
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=Filter(must=must_conditions),
                limit=limit,
                score_threshold=score_threshold,
            )

            return [
                {
                    "id": str(hit.id),
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": hit.payload.get("metadata", {}),
                    "timestamp": hit.payload.get("timestamp"),
                }
                for hit in results
            ]

        return await asyncio.to_thread(_sync_search)

    async def delete_memory(self, point_id: str) -> bool:
        """Delete a single memory by point ID."""
        def _sync_delete():
            from qdrant_client.models import PointIdsList
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[point_id]),
            )
            return True
        try:
            return await asyncio.to_thread(_sync_delete)
        except Exception as exc:
            logger.error("Failed to delete point %s: %s", point_id, exc)
            return False

    async def delete_user_memories(self, user_id: str) -> int:
        """
        Delete all memories for a user using filter-based deletion.
        Single atomic call — no manual pagination needed.

        Returns:
            Approximate count of deleted points (Qdrant doesn't always return exact count)
        """
        def _sync_delete():
            from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

            # Count first for logging purposes
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )
            count = count_result.count

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                    )
                ),
            )
            logger.info("Deleted %d memories for user %s", count, user_id)
            return count

        try:
            return await asyncio.to_thread(_sync_delete)
        except Exception as exc:
            logger.error("Failed to delete memories for user %s: %s", user_id, exc)
            return 0

    async def count_user_memories(self, user_id: str) -> int:
        """Count total memories stored for a user."""
        def _sync_count():
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )
            return result.count
        try:
            return await asyncio.to_thread(_sync_count)
        except Exception as exc:
            logger.error("Failed to count memories for user %s: %s", user_id, exc)
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant connectivity and return status detail."""
        def _sync_check():
            import time
            start = time.time()
            collections = self.client.get_collections()
            latency_ms = round((time.time() - start) * 1000, 1)
            return {
                "healthy": True,
                "latency_ms": latency_ms,
                "collections": len(collections.collections),
            }
        try:
            return await asyncio.to_thread(_sync_check)
        except Exception as exc:
            logger.error("Qdrant health check failed: %s", exc)
            return {"healthy": False, "error": str(exc)}


# ===========================================================================
# Singleton
# ===========================================================================
_wrapper_instance: Optional[AsyncQdrantWrapper] = None


def get_qdrant_wrapper() -> AsyncQdrantWrapper:
    """Get or create the global AsyncQdrantWrapper singleton."""
    global _wrapper_instance
    if _wrapper_instance is None:
        _wrapper_instance = AsyncQdrantWrapper()
    return _wrapper_instance


# Backward compatible singleton
qdrant = get_qdrant_wrapper()