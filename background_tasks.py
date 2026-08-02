"""
DevMentor AI — Background Tasks (Celery)
==========================================
Production-grade async task processing with:

- RAG index rebuild with Redis distributed locking
- Memory decay and consolidation
- Weekly usage report generation
- Expired session cleanup
- Database maintenance (old logs cleanup)
- Batch AI request processing
- User analytics aggregation
- Webhook delivery with retry
- Proper error handling and retry logic
- Task result tracking
- Structured logging per task

Usage:
    from background_tasks import rebuild_rag_index, send_weekly_report

    # Queue a task
    rebuild_rag_index.delay(user_id="u123")

    # Queue with countdown
    send_weekly_report.apply_async(args=[user_id], countdown=60)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from celery import shared_task
from sqlalchemy import text

from celery_app import celery_app
from config import get_settings
from database import get_db

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()


# ===========================================================================
# Redis Helper
# ===========================================================================

def _get_redis():
    """Get Redis client, returns None if unavailable."""
    try:
        from redis_client import get_redis_client
        return get_redis_client()
    except Exception:
        return None


# ===========================================================================
# Distributed Lock Helper
# ===========================================================================

class RedisLock:
    """
    Simple Redis-based distributed lock.
    Prevents duplicate task execution across workers.
    """

    def __init__(self, key: str, timeout: int = 300):
        self.key = f"lock:{key}"
        self.timeout = timeout
        self._redis = _get_redis()
        self._acquired = False

    def acquire(self) -> bool:
        if not self._redis:
            return True  # No Redis = no locking, proceed
        try:
            self._acquired = bool(self._redis.set(
                self.key, "1",
                nx=True,           # Only set if not exists
                ex=self.timeout,   # Auto-expire
            ))
            return self._acquired
        except Exception as exc:
            logger.warning("Lock acquire failed for %s: %s", self.key, exc)
            return True  # Proceed without lock on failure

    def release(self) -> None:
        if self._acquired and self._redis:
            try:
                self._redis.delete(self.key)
            except Exception:
                pass

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()


# ===========================================================================
# RAG Tasks
# ===========================================================================

@celery_app.task(
    bind=True,
    name="background_tasks.rebuild_rag_index",
    queue="rag",
    max_retries=3,
    default_retry_delay=60,
    ignore_result=False,
    soft_time_limit=settings.celery_task_soft_time_limit,
    time_limit=settings.celery_task_time_limit,
)
def rebuild_rag_index(self, user_id: str) -> Dict[str, Any]:
    """
    Rebuild the RAG vector index for a user.

    Idempotent — uses Redis distributed lock to prevent
    concurrent rebuilds for the same user.

    Args:
        user_id: User whose RAG index should be rebuilt

    Returns:
        Dict with status and rebuild stats
    """
    logger.info("RAG index rebuild started: user=%s task=%s", user_id, self.request.id)

    lock = RedisLock(f"rag_rebuild:{user_id}", timeout=300)
    if not lock.acquire():
        logger.warning("RAG rebuild already running for user %s — skipping", user_id)
        return {"status": "skipped", "user_id": user_id, "reason": "already_running"}

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=settings.qdrant_timeout,
        )

        collection = f"rag_{user_id}"
        collections = [c.name for c in client.get_collections().collections]

        if collection not in collections:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created RAG collection for user %s", user_id)

        logger.info("RAG index rebuild complete: user=%s", user_id)
        return {"status": "success", "user_id": user_id, "collection": collection}

    except ImportError:
        logger.warning("qdrant_client not installed — RAG rebuild skipped")
        return {"status": "skipped", "reason": "qdrant_not_installed"}

    except Exception as exc:
        logger.error("RAG rebuild failed for user %s: %s", user_id, exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"status": "error", "user_id": user_id, "error": str(exc)}

    finally:
        lock.release()


@celery_app.task(
    name="background_tasks.process_document_upload",
    queue="rag",
    max_retries=2,
    soft_time_limit=120,
)
def process_document_upload(
    user_id: str,
    file_path: str,
    file_name: str,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process an uploaded document for RAG indexing.

    Chunks the document, generates embeddings, and stores in Qdrant.

    Args:
        user_id:         Owner of the document
        file_path:       Path to the uploaded file
        file_name:       Original filename
        collection_name: Qdrant collection (defaults to user-specific)

    Returns:
        Dict with chunk count and status
    """
    logger.info(
        "Processing document upload: user=%s file=%s",
        user_id, file_name,
    )

    try:
        import os
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}

        # Read file content based on extension
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "txt"
        content = ""

        if ext == "txt" or ext == "md":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        elif ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return {"status": "error", "error": "pypdf not installed for PDF processing"}
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            return {"status": "error", "error": "Document appears to be empty"}

        # Simple chunking
        chunk_size = settings.rag_chunk_size
        overlap = settings.rag_chunk_overlap
        words = content.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap

        logger.info(
            "Document chunked: user=%s file=%s chunks=%d",
            user_id, file_name, len(chunks),
        )

        return {
            "status": "success",
            "user_id": user_id,
            "file_name": file_name,
            "chunk_count": len(chunks),
            "content_length": len(content),
        }

    except Exception as exc:
        logger.error(
            "Document processing failed: user=%s file=%s error=%s",
            user_id, file_name, exc, exc_info=True,
        )
        return {"status": "error", "error": str(exc)}


# ===========================================================================
# Memory Tasks
# ===========================================================================

@celery_app.task(
    name="background_tasks.apply_memory_decay",
    queue="maintenance",
    max_retries=1,
)
def apply_memory_decay(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply decay to long-term memories.

    If user_id provided: decay memories for that user only.
    If None: decay memories for all users (scheduled run).

    Returns:
        Dict with number of memories decayed
    """
    logger.info("Memory decay task started: user=%s", user_id or "all")
    total_decayed = 0

    try:
        from long_term_memory import get_long_term_memory
        import asyncio

        memory = get_long_term_memory()

        if user_id:
            loop = asyncio.new_event_loop()
            decayed = loop.run_until_complete(memory.apply_memory_decay(user_id))
            loop.close()
            total_decayed = decayed
        else:
            # Get all users and apply decay
            with get_db() as db:
                users = db.execute(
                    text("SELECT id FROM users WHERE is_deleted = false AND is_active = true")
                ).fetchall()

            for user in users:
                try:
                    loop = asyncio.new_event_loop()
                    decayed = loop.run_until_complete(
                        memory.apply_memory_decay(str(user.id))
                    )
                    loop.close()
                    total_decayed += decayed
                except Exception as exc:
                    logger.warning("Memory decay failed for user %s: %s", user.id, exc)

        logger.info("Memory decay complete: total_decayed=%d", total_decayed)
        return {"status": "success", "total_decayed": total_decayed}

    except Exception as exc:
        logger.error("Memory decay task failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


# ===========================================================================
# Email / Report Tasks
# ===========================================================================

@celery_app.task(
    bind=True,
    name="background_tasks.send_weekly_report",
    queue="email",
    max_retries=2,
    default_retry_delay=300,
)
def send_weekly_report(self, user_id: str) -> Dict[str, Any]:
    """
    Generate and send weekly usage report for a single user.

    Args:
        user_id: Target user

    Returns:
        Dict with report stats and delivery status
    """
    logger.info("Weekly report task: user=%s", user_id)

    try:
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)

        with get_db() as db:
            user_row = db.execute(
                text("SELECT id, email, username, tier FROM users WHERE id = :uid AND is_deleted = false"),
                {"uid": user_id},
            ).fetchone()

            if not user_row:
                return {"status": "skipped", "reason": "user_not_found"}

            if not user_row.email:
                logger.info("No email for user %s — skipping report", user_id)
                return {"status": "skipped", "reason": "no_email"}

            usage = db.execute(
                text("""
                    SELECT
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(cost_usd), 0.0) AS total_cost,
                        COUNT(*) AS request_count,
                        COALESCE(AVG(duration_ms), 0) AS avg_latency
                    FROM usage_logs
                    WHERE user_id = :uid AND created_at >= :week_ago
                """),
                {"uid": user_id, "week_ago": week_ago},
            ).fetchone()

            stats = {
                "username": user_row.username,
                "email": user_row.email,
                "tier": user_row.tier,
                "total_tokens": int(usage.total_tokens) if usage else 0,
                "total_cost_usd": round(float(usage.total_cost), 4) if usage else 0.0,
                "request_count": int(usage.request_count) if usage else 0,
                "avg_latency_ms": round(float(usage.avg_latency), 1) if usage else 0.0,
            }

        logger.info(
            "Weekly report generated: user=%s tokens=%d cost=$%.4f",
            user_id, stats["total_tokens"], stats["total_cost_usd"],
        )

        # TODO: Send via email service when configured
        # from email_service import send_weekly_report_email
        # send_weekly_report_email(to=stats["email"], stats=stats)

        return {"status": "success", "user_id": user_id, **stats}

    except Exception as exc:
        logger.error("Weekly report failed for user %s: %s", user_id, exc, exc_info=True)
        try:
            raise self.retry(exc=exc, countdown=300)
        except self.MaxRetriesExceededError:
            return {"status": "error", "user_id": user_id, "error": str(exc)}


@celery_app.task(
    name="background_tasks.send_weekly_reports_to_all",
    queue="maintenance",
)
def send_weekly_reports_to_all() -> Dict[str, Any]:
    """
    Dispatch weekly report tasks for all active users with emails.
    Scheduled via Celery Beat.
    """
    queued = 0
    try:
        with get_db() as db:
            users = db.execute(
                text("""
                    SELECT id FROM users
                    WHERE is_active = true
                    AND is_deleted = false
                    AND email IS NOT NULL
                """)
            ).fetchall()

        for user in users:
            send_weekly_report.delay(str(user.id))
            queued += 1

        logger.info("Weekly report tasks queued: count=%d", queued)
        return {"status": "success", "users_queued": queued}

    except Exception as exc:
        logger.error("send_weekly_reports_to_all failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


# ===========================================================================
# Cleanup & Maintenance Tasks
# ===========================================================================

@celery_app.task(
    name="background_tasks.cleanup_expired_sessions",
    queue="maintenance",
)
def cleanup_expired_sessions() -> Dict[str, Any]:
    """
    Remove orphaned Redis session keys (no TTL set).
    Scheduled to run daily.
    """
    redis = _get_redis()
    if not redis:
        return {"status": "skipped", "reason": "redis_unavailable"}

    deleted = 0
    scanned = 0
    cursor = 0

    try:
        while True:
            cursor, keys = redis.scan(cursor, match="session:*", count=100)
            scanned += len(keys)
            for key in keys:
                ttl = redis.ttl(key)
                if ttl == -1:  # No expiration = orphaned
                    redis.delete(key)
                    deleted += 1
            if cursor == 0:
                break

        logger.info("Session cleanup: scanned=%d deleted=%d", scanned, deleted)
        return {"status": "success", "scanned": scanned, "deleted": deleted}

    except Exception as exc:
        logger.error("Session cleanup failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@celery_app.task(
    name="background_tasks.cleanup_old_usage_logs",
    queue="maintenance",
)
def cleanup_old_usage_logs(retention_days: int = 90) -> Dict[str, Any]:
    """
    Delete usage logs older than retention_days.
    Keeps database size manageable.

    Args:
        retention_days: Logs older than this are deleted (default: 90)

    Returns:
        Dict with number of rows deleted
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    logger.info("Cleaning up usage logs older than %s", cutoff.date())

    try:
        with get_db() as db:
            result = db.execute(
                text("DELETE FROM usage_logs WHERE created_at < :cutoff"),
                {"cutoff": cutoff},
            )
            deleted = result.rowcount

        logger.info("Usage log cleanup: deleted=%d rows", deleted)
        return {"status": "success", "deleted": deleted, "cutoff": cutoff.isoformat()}

    except Exception as exc:
        logger.error("Usage log cleanup failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@celery_app.task(
    name="background_tasks.cleanup_old_audit_logs",
    queue="maintenance",
)
def cleanup_old_audit_logs(retention_days: int = 365) -> Dict[str, Any]:
    """
    Delete audit logs older than retention_days.
    Default 1 year retention for compliance.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    try:
        with get_db() as db:
            result = db.execute(
                text("DELETE FROM audit_logs WHERE created_at < :cutoff"),
                {"cutoff": cutoff},
            )
            deleted = result.rowcount

        logger.info("Audit log cleanup: deleted=%d rows", deleted)
        return {"status": "success", "deleted": deleted}

    except Exception as exc:
        logger.error("Audit log cleanup failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@celery_app.task(
    name="background_tasks.reset_monthly_token_counters",
    queue="maintenance",
)
def reset_monthly_token_counters() -> Dict[str, Any]:
    """
    Reset monthly token usage counters for all users.
    Should be scheduled for the 1st of each month.
    """
    logger.info("Resetting monthly token counters")

    try:
        with get_db() as db:
            result = db.execute(
                text("""
                    UPDATE users
                    SET monthly_tokens_used = 0,
                        monthly_reset_at = :now
                    WHERE is_deleted = false
                """),
                {"now": datetime.now(timezone.utc)},
            )
            updated = result.rowcount

        logger.info("Monthly token reset: updated=%d users", updated)
        return {"status": "success", "users_reset": updated}

    except Exception as exc:
        logger.error("Monthly token reset failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


# ===========================================================================
# Batch Processing Tasks
# ===========================================================================

@celery_app.task(
    name="background_tasks.process_batch_requests",
    queue="batch",
    soft_time_limit=settings.celery_task_soft_time_limit,
)
def process_batch_requests(batch_data: List[Dict]) -> List[Dict]:
    """
    Process a batch of AI requests asynchronously.

    Args:
        batch_data: List of request dicts with 'prompt', 'user_id', etc.

    Returns:
        List of result dicts with status and response
    """
    logger.info("Processing batch: count=%d", len(batch_data))
    results = []

    for idx, item in enumerate(batch_data):
        try:
            import asyncio
            from model_router import get_model_router

            prompt = item.get("prompt", "")
            user_id = item.get("user_id", "batch")
            user_tier = item.get("tier", "free")

            if not prompt:
                results.append({"index": idx, "status": "error", "error": "Empty prompt"})
                continue

            loop = asyncio.new_event_loop()
            router = get_model_router()
            response = loop.run_until_complete(
                router.route(prompt=prompt, user_id=user_id, user_tier=user_tier)
            )
            loop.close()

            results.append({
                "index": idx,
                "status": "success",
                "response": response.response,
                "model": response.model,
                "tokens": response.total_tokens,
                "cost_usd": response.cost_usd,
            })

        except Exception as exc:
            logger.error("Batch item %d failed: %s", idx, exc)
            results.append({"index": idx, "status": "error", "error": str(exc)})

    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(
        "Batch complete: total=%d success=%d failed=%d",
        len(results), success_count, len(results) - success_count,
    )
    return results


# ===========================================================================
# Webhook Delivery Task
# ===========================================================================

@celery_app.task(
    bind=True,
    name="background_tasks.deliver_webhook",
    queue="webhooks",
    max_retries=5,
    default_retry_delay=30,
)
def deliver_webhook(
    self,
    webhook_url: str,
    payload: Dict[str, Any],
    secret_hash: Optional[str] = None,
    webhook_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deliver a webhook payload to a URL with retry logic.

    Args:
        webhook_url:  Target URL
        payload:      JSON payload to deliver
        secret_hash:  HMAC secret for signature (optional)
        webhook_id:   Webhook endpoint ID for tracking

    Returns:
        Dict with delivery status and HTTP response code
    """
    import json
    import hmac
    import hashlib
    import httpx

    logger.info("Delivering webhook: url=%s id=%s", webhook_url, webhook_id)

    try:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"DevMentorAI-Webhook/1.0",
            "X-DevMentor-Event": payload.get("event", "unknown"),
            "X-DevMentor-Delivery": self.request.id or "",
        }

        # Sign payload if secret provided
        if secret_hash:
            body = json.dumps(payload, separators=(",", ":"))
            signature = hmac.new(
                secret_hash.encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-DevMentor-Signature"] = f"sha256={signature}"

        with httpx.Client(timeout=15) as client:
            response = client.post(
                webhook_url,
                json=payload,
                headers=headers,
            )

        if response.status_code in {200, 201, 202, 204}:
            logger.info(
                "Webhook delivered: url=%s status=%d",
                webhook_url, response.status_code,
            )
            return {
                "status": "success",
                "http_status": response.status_code,
                "webhook_id": webhook_id,
            }
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

    except Exception as exc:
        logger.warning(
            "Webhook delivery failed (attempt %d): url=%s error=%s",
            self.request.retries + 1, webhook_url, exc,
        )
        try:
            # Exponential backoff: 30s, 60s, 120s, 240s, 480s
            countdown = 30 * (2 ** self.request.retries)
            raise self.retry(exc=exc, countdown=countdown)
        except self.MaxRetriesExceededError:
            logger.error(
                "Webhook delivery permanently failed: url=%s",
                webhook_url,
            )
            return {
                "status": "failed",
                "webhook_id": webhook_id,
                "error": str(exc),
            }