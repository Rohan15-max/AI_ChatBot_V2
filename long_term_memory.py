"""
DevMentor AI — Long-Term Memory Service
=========================================
Production-grade persistent memory with:

- Qdrant vector store for semantic memory search
- SQLAlchemy for structured memory metadata
- Memory types: factual, procedural, emotional, preference
- Importance scoring and decay
- Memory consolidation (merge similar memories)
- User profile building from memories
- GDPR-compliant deletion
- Memory extraction from conversations via LLM
- Async operations throughout
- Graceful degradation on failures

UPGRADE NOTE (this revision):

`_recall_from_qdrant` is a *synchronous* method, invoked via
`asyncio.to_thread(...)` from `recall_relevant` — meaning it actually runs
in a worker thread, not on the event loop. The previous version tried to
schedule background access-count tracking from inside that sync method via:

    asyncio.create_task(...) if asyncio.get_event_loop().is_running() else None

`asyncio.get_event_loop()` called from a non-main thread with no loop set
raises `RuntimeError: There is no current event loop in thread '...'` on
Python 3.10+ — to_thread workers don't have one by default. So every
successful memory recall would throw right after fetching results.

Fixed by moving the access-count update out of the sync thread-pool method
entirely. `_recall_from_qdrant` now just returns the recalled memories;
`recall_relevant` (the actual async method, which genuinely does run on the
event loop) awaits the access-count update itself afterward. Functionally
identical outcome, just scheduled from a context where asyncio actually
has a loop to work with.

How it works:
    1. After each conversation, extract key facts/preferences
    2. Store as vector embeddings in Qdrant
    3. On each new conversation, retrieve relevant memories
    4. Inject memories into system prompt for context

Usage:
    from long_term_memory import LongTermMemory

    memory = LongTermMemory()

    # Store after conversation
    await memory.store_conversation(user_id, messages)

    # Retrieve before LLM call
    memories = await memory.recall_relevant(user_id, query, limit=5)
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from config import get_settings
from database import LongTermMemory as MemoryModel, get_db, utc_now

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
MEMORY_COLLECTION = "long_term_memory"
MAX_MEMORY_TEXT_LENGTH = 10000
MIN_IMPORTANCE_SCORE = 0.1
DEFAULT_IMPORTANCE = 0.5


# ===========================================================================
# Embedding Model (singleton)
# ===========================================================================

@lru_cache(maxsize=1)
def _get_embedding_model():
    """
    Load and cache the sentence transformer model.
    Called once on first use — lazy loading avoids startup delay.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        logger.info("Memory embedding model loaded: %s", model_name)
        return model
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        return None


def _generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for text. Returns None on failure."""
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        text = text[:2048]  # Truncate for embedding
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        return None


# ===========================================================================
# Qdrant Helper
# ===========================================================================

def _get_qdrant():
    """Get Qdrant client. Returns None if unavailable."""
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
        logger.warning("Qdrant unavailable for memory: %s", exc)
        return None


def _ensure_memory_collection() -> None:
    """Create Qdrant memory collection if it doesn't exist."""
    qdrant = _get_qdrant()
    if not qdrant:
        return
    try:
        from qdrant_client.models import Distance, VectorParams
        collections = [c.name for c in qdrant.get_collections().collections]
        if MEMORY_COLLECTION not in collections:
            qdrant.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created memory collection: %s", MEMORY_COLLECTION)
    except Exception as exc:
        logger.error("Failed to ensure memory collection: %s", exc)


# ===========================================================================
# Memory Data Classes
# ===========================================================================

@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    user_id: str
    content: str
    memory_type: str              # factual | procedural | emotional | preference
    importance_score: float
    created_at: str
    last_accessed: Optional[str]
    access_count: int
    decay_score: float
    source_thread_id: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecalledMemory:
    """A memory retrieved during recall."""
    content: str
    memory_type: str
    similarity_score: float
    importance_score: float
    created_at: str
    memory_id: str


# ===========================================================================
# Memory Extraction
# ===========================================================================

async def extract_memories_from_conversation(
    messages: List[Dict],
    user_id: str,
) -> List[Dict[str, Any]]:
    """
    Use LLM to extract memorable facts from a conversation.

    Extracts:
    - User preferences (likes/dislikes)
    - Personal facts (name, location, job)
    - Skills and expertise
    - Goals and objectives
    - Important decisions made

    Args:
        messages: Conversation messages
        user_id:  User ID for context

    Returns:
        List of extracted memory dicts with content and type
    """
    if not messages:
        return []

    conv_text = "\n".join([
        f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')[:500]}"
        for msg in messages[-20:]  # Last 20 messages
        if msg.get('content')
    ])

    if len(conv_text) < 50:
        return []

    extraction_prompt = f"""Analyze this conversation and extract memorable facts about the user.

Conversation:
{conv_text[:3000]}

Extract facts in these categories:
1. FACTUAL: Personal facts (name, age, location, job, education)
2. PREFERENCE: Likes, dislikes, favorite things
3. PROCEDURAL: How they like things done, their workflow
4. EMOTIONAL: Important feelings or experiences they shared

Format as JSON array:
[
  {{"content": "fact here", "type": "factual|preference|procedural|emotional", "importance": 0.1-1.0}},
  ...
]

Only extract clear, specific facts. Return empty array [] if nothing notable.
Return ONLY the JSON array, no other text."""

    try:
        from model_router import get_model_router
        router = get_model_router()
        response = await router.route(
            prompt=extraction_prompt,
            user_id=user_id,
            user_tier="free",  # Use cheap model for extraction
            preferred_model="gemini-flash",
        )

        import json
        text = response.response.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        extracted = json.loads(text)
        if isinstance(extracted, list):
            return [
                m for m in extracted
                if isinstance(m, dict)
                and m.get("content")
                and m.get("type") in {"factual", "preference", "procedural", "emotional"}
            ]

    except Exception as exc:
        logger.warning("Memory extraction failed: %s", exc)

    return []


# ===========================================================================
# Long-Term Memory Service
# ===========================================================================

class LongTermMemory:
    """
    Production-grade long-term memory service.

    Combines:
    - Qdrant for semantic similarity search
    - PostgreSQL (via SQLAlchemy) for structured metadata
    - LLM-powered memory extraction
    - Importance scoring and decay
    """

    def __init__(self):
        _ensure_memory_collection()

    # -----------------------------------------------------------------------
    # Store
    # -----------------------------------------------------------------------

    async def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "factual",
        importance_score: float = DEFAULT_IMPORTANCE,
        source_thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a single memory.

        Args:
            user_id:          User this memory belongs to
            content:          Memory text content
            memory_type:      factual | preference | procedural | emotional
            importance_score: 0.0-1.0, higher = less likely to decay
            source_thread_id: Thread this memory came from
            metadata:         Additional metadata

        Returns:
            Memory ID if stored, None on failure
        """
        if not content or not content.strip():
            return None

        content = content.strip()[:MAX_MEMORY_TEXT_LENGTH]
        memory_id = str(uuid.uuid4())

        db_success = await asyncio.to_thread(
            self._store_in_db,
            memory_id, user_id, content, memory_type,
            importance_score, source_thread_id,
        )

        qdrant_success = await asyncio.to_thread(
            self._store_in_qdrant,
            memory_id, user_id, content, memory_type,
            importance_score, source_thread_id, metadata or {},
        )

        if db_success or qdrant_success:
            logger.info(
                "Memory stored: user=%s type=%s importance=%.2f",
                user_id, memory_type, importance_score,
            )
            return memory_id

        return None

    def _store_in_db(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        importance_score: float,
        source_thread_id: Optional[str],
    ) -> bool:
        """Store memory metadata in PostgreSQL."""
        try:
            with get_db() as db:
                memory = MemoryModel(
                    id=memory_id,
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    importance_score=importance_score,
                    source_thread_id=source_thread_id,
                    decay_score=1.0,
                    access_count=0,
                )
                db.add(memory)
            return True
        except Exception as exc:
            logger.error("DB memory store failed: %s", exc)
            return False

    def _store_in_qdrant(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        importance_score: float,
        source_thread_id: Optional[str],
        metadata: Dict,
    ) -> bool:
        """Store memory vector in Qdrant."""
        qdrant = _get_qdrant()
        if not qdrant:
            return False

        embedding = _generate_embedding(content)
        if embedding is None:
            return False

        try:
            from qdrant_client.models import PointStruct
            qdrant.upsert(
                collection_name=MEMORY_COLLECTION,
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "user_id": user_id,
                        "content": content,
                        "memory_type": memory_type,
                        "importance_score": importance_score,
                        "source_thread_id": source_thread_id,
                        "created_at": utc_now().isoformat(),
                        "decay_score": 1.0,
                        **metadata,
                    },
                )],
            )
            return True
        except Exception as exc:
            logger.error("Qdrant memory store failed: %s", exc)
            return False

    async def store_conversation(
        self,
        user_id: str,
        messages: List[Dict],
        thread_id: Optional[str] = None,
        extract_facts: bool = True,
    ) -> List[str]:
        """
        Process and store memories from a conversation.

        Args:
            user_id:      User ID
            messages:     Conversation messages
            thread_id:    Source thread ID
            extract_facts: Whether to use LLM to extract facts

        Returns:
            List of stored memory IDs
        """
        if not messages or not settings.feature_long_term_memory:
            return []

        stored_ids = []

        conv_text = "\n".join([
            f"{m.get('role', '')}: {m.get('content', '')[:300]}"
            for m in messages
            if m.get('content')
        ])

        if conv_text:
            summary_id = await self.store_memory(
                user_id=user_id,
                content=conv_text[:MAX_MEMORY_TEXT_LENGTH],
                memory_type="factual",
                importance_score=0.4,
                source_thread_id=thread_id,
                metadata={"subtype": "conversation_summary", "message_count": len(messages)},
            )
            if summary_id:
                stored_ids.append(summary_id)

        if extract_facts:
            try:
                extracted = await extract_memories_from_conversation(messages, user_id)
                for fact in extracted[:10]:  # Max 10 facts per conversation
                    mem_id = await self.store_memory(
                        user_id=user_id,
                        content=fact["content"],
                        memory_type=fact.get("type", "factual"),
                        importance_score=float(fact.get("importance", 0.5)),
                        source_thread_id=thread_id,
                    )
                    if mem_id:
                        stored_ids.append(mem_id)
            except Exception as exc:
                logger.warning("Fact extraction failed: %s", exc)

        logger.info(
            "Stored %d memories for user %s from conversation",
            len(stored_ids), user_id,
        )
        return stored_ids

    # -----------------------------------------------------------------------
    # Recall
    # -----------------------------------------------------------------------

    async def recall_relevant(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        min_score: float = 0.5,
    ) -> List[RecalledMemory]:
        """
        Retrieve memories relevant to a query.

        Args:
            user_id:      User ID to retrieve memories for
            query:        Query to find relevant memories
            limit:        Max memories to return
            memory_types: Filter by type (factual/preference/etc.)
            min_score:    Minimum similarity score

        Returns:
            List of RecalledMemory sorted by relevance

        Note: access-count tracking for whatever memories come back is
        fired here, on the event loop, rather than from inside the
        thread-pool search itself — see module docstring for why.
        """
        if not query or not settings.feature_long_term_memory:
            return []

        memories = await asyncio.to_thread(
            self._recall_from_qdrant,
            user_id, query, limit, memory_types, min_score,
        )

        if memories:
            # Fire-and-forget: tracking access counts should never block
            # or fail the actual recall result the caller is waiting on.
            asyncio.create_task(
                self._update_access_counts(user_id, [m.memory_id for m in memories])
            )

        return memories

    def _recall_from_qdrant(
        self,
        user_id: str,
        query: str,
        limit: int,
        memory_types: Optional[List[str]],
        min_score: float,
    ) -> List[RecalledMemory]:
        """
        Synchronous Qdrant search, wrapped for the thread pool by the caller.

        This method does ONLY the search and result-shaping — no asyncio
        calls of any kind. It runs in a worker thread with no event loop
        attached, so anything async-related belongs in the calling
        coroutine (recall_relevant) instead, after to_thread() returns.
        """
        qdrant = _get_qdrant()
        if not qdrant:
            return []

        embedding = _generate_embedding(query)
        if embedding is None:
            return []

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

            must_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            if memory_types:
                must_conditions.append(
                    FieldCondition(key="memory_type", match=MatchAny(any=memory_types))
                )

            search_filter = Filter(must=must_conditions)

            hits = qdrant.search(
                collection_name=MEMORY_COLLECTION,
                query_vector=embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=min_score,
                with_payload=True,
            )

            memories = []
            for hit in hits:
                payload = hit.payload or {}
                memories.append(RecalledMemory(
                    content=payload.get("content", ""),
                    memory_type=payload.get("memory_type", "factual"),
                    similarity_score=hit.score,
                    importance_score=payload.get("importance_score", 0.5),
                    created_at=payload.get("created_at", ""),
                    memory_id=str(hit.id),
                ))

            return memories

        except Exception as exc:
            logger.error("Qdrant memory recall failed: %s", exc)
            return []

    async def _update_access_counts(self, user_id: str, memory_ids: List[str]) -> None:
        """Update access tracking for recalled memories."""
        if not memory_ids:
            return
        try:
            with get_db() as db:
                memories = db.query(MemoryModel).filter(
                    MemoryModel.id.in_(memory_ids),
                    MemoryModel.user_id == user_id,
                ).all()
                for memory in memories:
                    memory.record_access()
        except Exception as exc:
            logger.debug("Access count update failed (non-critical): %s", exc)

    def format_memories_for_prompt(
        self,
        memories: List[RecalledMemory],
        max_chars: int = 2000,
    ) -> str:
        """
        Format recalled memories for injection into LLM system prompt.

        Args:
            memories:  List of recalled memories
            max_chars: Max total characters

        Returns:
            Formatted string ready for system prompt injection
        """
        if not memories:
            return ""

        lines = ["## What I Remember About You"]
        total_chars = len(lines[0])

        for mem in memories:
            line = f"- [{mem.memory_type}] {mem.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # User Profile
    # -----------------------------------------------------------------------

    async def get_user_profile(
        self,
        user_id: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Build a user profile from their memories.

        Returns:
            Dict with categorized memories and metadata
        """
        profile: Dict[str, Any] = {
            "user_id": user_id,
            "facts": [],
            "preferences": [],
            "procedures": [],
            "emotional": [],
            "total_memories": 0,
            "last_updated": utc_now().isoformat(),
        }

        try:
            with get_db() as db:
                all_memories = (
                    db.query(MemoryModel)
                    .filter(
                        MemoryModel.user_id == user_id,
                        MemoryModel.is_deleted == False,
                    )
                    .order_by(MemoryModel.importance_score.desc())
                    .limit(limit)
                    .all()
                )

                profile["total_memories"] = len(all_memories)

                for mem in all_memories:
                    entry = {"content": mem.content, "importance": mem.importance_score}
                    if mem.memory_type == "factual":
                        profile["facts"].append(entry)
                    elif mem.memory_type == "preference":
                        profile["preferences"].append(entry)
                    elif mem.memory_type == "procedural":
                        profile["procedures"].append(entry)
                    elif mem.memory_type == "emotional":
                        profile["emotional"].append(entry)

        except Exception as exc:
            logger.error("Failed to build user profile: %s", exc)

        return profile

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------

    async def apply_memory_decay(self, user_id: str) -> int:
        """
        Apply time-based decay to memories.

        Memories that haven't been accessed decay over time.
        Low-importance, decayed memories are soft-deleted.

        Returns:
            Number of memories decayed/removed
        """
        return await asyncio.to_thread(self._apply_decay_sync, user_id)

    def _apply_decay_sync(self, user_id: str) -> int:
        """Synchronous decay application."""
        decayed_count = 0
        cutoff_date = utc_now() - timedelta(days=settings.memory_decay_days)

        try:
            with get_db() as db:
                old_memories = (
                    db.query(MemoryModel)
                    .filter(
                        MemoryModel.user_id == user_id,
                        MemoryModel.created_at < cutoff_date,
                        MemoryModel.importance_score < 0.7,
                        MemoryModel.is_deleted == False,
                    )
                    .all()
                )

                for mem in old_memories:
                    days_old = (utc_now() - mem.created_at).days
                    decay = max(0.0, 1.0 - (days_old / (settings.memory_decay_days * 2)))
                    mem.decay_score = decay

                    if decay < 0.1 and mem.importance_score < MIN_IMPORTANCE_SCORE:
                        mem.soft_delete()
                        decayed_count += 1

            logger.info(
                "Memory decay applied for user %s: %d removed",
                user_id, decayed_count,
            )
        except Exception as exc:
            logger.error("Memory decay failed for user %s: %s", user_id, exc)

        return decayed_count

    async def delete_user_data(self, user_id: str) -> int:
        """
        Delete ALL memories for a user (GDPR right to erasure).

        Args:
            user_id: User whose data should be deleted

        Returns:
            Number of memories deleted
        """
        deleted_count = 0

        try:
            with get_db() as db:
                memories = db.query(MemoryModel).filter(
                    MemoryModel.user_id == user_id
                ).all()
                count = len(memories)
                for mem in memories:
                    db.delete(mem)
                deleted_count += count
        except Exception as exc:
            logger.error("DB memory deletion failed for user %s: %s", user_id, exc)

        await asyncio.to_thread(self._delete_from_qdrant, user_id)

        logger.info("Deleted %d memories for user %s (GDPR)", deleted_count, user_id)
        return deleted_count

    def _delete_from_qdrant(self, user_id: str) -> None:
        """Delete all user points from Qdrant."""
        qdrant = _get_qdrant()
        if not qdrant:
            return

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
            qdrant.delete(
                collection_name=MEMORY_COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                    )
                ),
            )
        except Exception as exc:
            logger.error("Qdrant memory deletion failed for user %s: %s", user_id, exc)

    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user."""
        try:
            with get_db() as db:
                total = db.query(MemoryModel).filter(
                    MemoryModel.user_id == user_id,
                    MemoryModel.is_deleted == False,
                ).count()

                by_type = {}
                for mem_type in ["factual", "preference", "procedural", "emotional"]:
                    count = db.query(MemoryModel).filter(
                        MemoryModel.user_id == user_id,
                        MemoryModel.memory_type == mem_type,
                        MemoryModel.is_deleted == False,
                    ).count()
                    by_type[mem_type] = count

                return {
                    "user_id": user_id,
                    "total_memories": total,
                    "by_type": by_type,
                    "limit": settings.memory_max_long_term,
                    "usage_percent": round(total / settings.memory_max_long_term * 100, 1),
                }
        except Exception as exc:
            logger.error("Memory stats failed: %s", exc)
            return {"user_id": user_id, "total_memories": 0, "error": str(exc)}


# ===========================================================================
# Singleton
# ===========================================================================
_memory_instance: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    """Get or create the global LongTermMemory singleton."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LongTermMemory()
    return _memory_instance


# Backward compatible singleton
long_term_memory = get_long_term_memory()