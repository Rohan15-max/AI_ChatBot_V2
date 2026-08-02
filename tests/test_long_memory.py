"""
Tests for long_term_memory.py's LongTermMemory service.

REWRITE NOTE: the original test imported `from long_term_memory import
MemoryStore` and called sync methods `.add()`, `.decay_memories()`,
`.retrieve()`. None of these exist. The real class is `LongTermMemory`
(accessed via the `get_long_term_memory()` singleton), every public method
is `async def`, and the actual method names are `store_memory()`,
`apply_memory_decay()`, and `recall_relevant()`.

These tests require a real Qdrant connection to meaningfully exercise
recall (semantic search needs actual vector storage) — they're written to
skip gracefully rather than false-pass if Qdrant isn't reachable, mirroring
the same principle as test_rate_limit.py's Redis-skip fixture: a test that
silently does nothing because its dependency is down is worse than no
test, since it looks green for the wrong reason.
"""

import uuid
import pytest

from long_term_memory import get_long_term_memory
from qdrant_client import get_qdrant_wrapper


@pytest.fixture
def memory():
    return get_long_term_memory()


@pytest.fixture(autouse=True)
async def require_qdrant():
    """Skip if Qdrant isn't reachable — recall_relevant needs a real vector store."""
    wrapper = get_qdrant_wrapper()
    health = await wrapper.health_check()
    if not health.get("healthy"):
        pytest.skip(f"Qdrant is not reachable: {health.get('error')}")


@pytest.fixture
def test_user_id():
    """Fresh UUID per test run so memories don't accumulate across runs."""
    return f"test_user_{uuid.uuid4().hex[:10]}"


@pytest.mark.asyncio
async def test_store_memory_returns_id(memory, test_user_id):
    """store_memory() should return a memory ID string on success."""
    memory_id = await memory.store_memory(
        user_id=test_user_id,
        content="The user prefers Python over JavaScript for backend work.",
        memory_type="preference",
        importance_score=0.7,
    )
    assert memory_id is not None
    assert isinstance(memory_id, str)


@pytest.mark.asyncio
async def test_recall_relevant_finds_stored_memory(memory, test_user_id):
    """
    A memory about a specific, distinctive fact should be retrievable by a
    semantically related query, even with different wording. min_score is
    lowered from recall_relevant()'s 0.5 default since test content is
    short and embeddings of short strings can score lower than real
    conversation-length content would.
    """
    await memory.store_memory(
        user_id=test_user_id,
        content="The user is building a fintech application using Python and PostgreSQL.",
        memory_type="factual",
        importance_score=0.8,
    )

    results = await memory.recall_relevant(
        user_id=test_user_id,
        query="What is the user building?",
        limit=5,
        min_score=0.3,
    )

    assert len(results) > 0
    assert any("fintech" in r.content.lower() for r in results)


@pytest.mark.asyncio
async def test_recall_relevant_scopes_to_user(memory):
    """A query for one user must never surface another user's memories."""
    user_a = f"test_user_a_{uuid.uuid4().hex[:8]}"
    user_b = f"test_user_b_{uuid.uuid4().hex[:8]}"

    await memory.store_memory(
        user_id=user_a,
        content="User A's secret project is called Aurora.",
        memory_type="factual",
        importance_score=0.9,
    )

    results = await memory.recall_relevant(
        user_id=user_b,
        query="What is the secret project called?",
        limit=5,
        min_score=0.1,
    )

    assert not any("aurora" in r.content.lower() for r in results)


@pytest.mark.asyncio
async def test_delete_user_data_removes_memories(memory, test_user_id):
    """delete_user_data() (GDPR erasure) should remove all of a user's memories."""
    await memory.store_memory(
        user_id=test_user_id,
        content="Temporary memory for deletion test.",
        memory_type="factual",
    )

    deleted_count = await memory.delete_user_data(test_user_id)
    assert deleted_count >= 1

    stats = await memory.get_memory_stats(test_user_id)
    assert stats["total_memories"] == 0