"""
DevMentor AI — Redis Client
=============================
Production-grade Redis client with:

- Both sync and async clients (sync for Celery/Flask, async for asyncio code)
- Connection pooling with health checks
- Automatic retry with exponential backoff
- Graceful degradation when Redis is unavailable
- JSON serialization helpers
- Sliding window rate limiting
- Session storage helpers
- Cache helpers with TTL
- Singleton pattern with lazy initialization

Usage:
    # Sync (for Flask routes, Celery tasks)
    from redis_client import get_redis_client
    redis = get_redis_client()
    redis.set("key", "value", ex=60)

    # Async (for asyncio code)
    from redis_client import get_async_redis_client
    redis = await get_async_redis_client()
    await redis.set("key", "value", ex=60)
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import redis as sync_redis
import redis.asyncio as async_redis
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError, RedisError, TimeoutError
from redis.retry import Retry

from config import get_settings

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()


# ===========================================================================
# Sync Redis Client (for Flask, Celery)
# ===========================================================================

class SyncRedisClient:
    """
    Synchronous Redis client wrapper.

    Used in Flask routes, Celery tasks, and other sync contexts.
    Includes connection pooling, retry logic, and graceful degradation.
    """

    def __init__(self):
        self._client: Optional[sync_redis.Redis] = None
        self._connected = False

    def _build_client(self) -> sync_redis.Redis:
        """Build Redis client with production-grade connection settings."""
        retry = Retry(ExponentialBackoff(cap=10, base=1), retries=3)

        return sync_redis.from_url(
            settings.redis_url,
            password=settings.redis_password.get_secret_value() if settings.redis_password else None,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
            socket_keepalive=True,
            retry=retry,
            retry_on_error=[ConnectionError, TimeoutError],
            retry_on_timeout=settings.redis_retry_on_timeout,
            health_check_interval=30,
            max_connections=settings.redis_max_connections,
        )

    def _ensure_connected(self) -> bool:
        """Ensure client is connected, reconnect if needed."""
        if self._client is None:
            try:
                self._client = self._build_client()
                self._client.ping()
                self._connected = True
                logger.info("Redis connected (sync)")
            except Exception as exc:
                logger.error("Redis connection failed: %s", exc)
                self._client = None
                self._connected = False
                return False
        return True

    @property
    def connected(self) -> bool:
        """Check current connection status."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def get_raw_client(self) -> Optional[sync_redis.Redis]:
        """Get the underlying redis.Redis client for advanced operations."""
        if self._ensure_connected():
            return self._client
        return None

    # -----------------------------------------------------------------------
    # Basic Operations
    # -----------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        """Set a key with optional expiry and NX (only-if-not-exists) flag."""
        if not self._ensure_connected():
            return False
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            result = self._client.set(key, value, ex=ex, nx=nx)
            return bool(result)
        except RedisError as exc:
            logger.error("Redis SET failed for key %s: %s", key, exc)
            return False

    def get(self, key: str, parse_json: bool = True) -> Optional[Any]:
        """Get a value by key, optionally parsing as JSON."""
        if not self._ensure_connected():
            return None
        try:
            value = self._client.get(key)
            if value is None:
                return None
            if parse_json:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return value
        except RedisError as exc:
            logger.error("Redis GET failed for key %s: %s", key, exc)
            return None

    def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns count deleted."""
        if not self._ensure_connected() or not keys:
            return 0
        try:
            return self._client.delete(*keys)
        except RedisError as exc:
            logger.error("Redis DELETE failed: %s", exc)
            return 0

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if not self._ensure_connected():
            return False
        try:
            return self._client.exists(key) > 0
        except RedisError:
            return False

    def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiry on an existing key."""
        if not self._ensure_connected():
            return False
        try:
            return bool(self._client.expire(key, ttl_seconds))
        except RedisError:
            return False

    def ttl(self, key: str) -> int:
        """Get remaining TTL for a key. Returns -1 if no expiry, -2 if missing."""
        if not self._ensure_connected():
            return -2
        try:
            return self._client.ttl(key)
        except RedisError:
            return -2

    def incrbyfloat(self, key: str, amount: float) -> Optional[float]:
        """Atomically increment a float value."""
        if not self._ensure_connected():
            return None
        try:
            return self._client.incrbyfloat(key, amount)
        except RedisError as exc:
            logger.error("Redis INCRBYFLOAT failed: %s", exc)
            return None

    def keys(self, pattern: str) -> list:
        """
        Find keys matching pattern.
        WARNING: Use sparingly in production — prefer scan() for large datasets.
        """
        if not self._ensure_connected():
            return []
        try:
            return self._client.keys(pattern)
        except RedisError:
            return []

    def scan(self, cursor: int = 0, match: str = "*", count: int = 100) -> Tuple[int, list]:
        """Scan keys incrementally — safe for production use on large datasets."""
        if not self._ensure_connected():
            return 0, []
        try:
            return self._client.scan(cursor=cursor, match=match, count=count)
        except RedisError:
            return 0, []

    def pipeline(self, transaction: bool = True):
        """Get a pipeline for batched operations."""
        if not self._ensure_connected():
            return None
        return self._client.pipeline(transaction=transaction)

    def eval(self, script: str, numkeys: int, *keys_and_args):
        """Execute a Lua script atomically."""
        if not self._ensure_connected():
            return None
        try:
            return self._client.eval(script, numkeys, *keys_and_args)
        except RedisError as exc:
            logger.error("Redis EVAL failed: %s", exc)
            return None

    def hincrby(self, key: str, field: str, amount: int = 1) -> Optional[int]:
        """Increment a hash field."""
        if not self._ensure_connected():
            return None
        try:
            return self._client.hincrby(key, field, amount)
        except RedisError:
            return None

    def hgetall(self, key: str) -> Dict:
        """Get all fields in a hash."""
        if not self._ensure_connected():
            return {}
        try:
            return self._client.hgetall(key) or {}
        except RedisError:
            return {}

    def setex(self, key: str, ttl_seconds: int, value: Any) -> bool:
        """Set a key with expiry."""
        return self.set(key, value, ex=ttl_seconds)

    def ping(self) -> bool:
        """Health check ping."""
        if not self._ensure_connected():
            return False
        try:
            return self._client.ping()
        except RedisError:
            return False

    # -----------------------------------------------------------------------
    # High-Level Helpers
    # -----------------------------------------------------------------------

    def cache_response(self, query_hash: str, response: Any, ttl: int = 3600) -> bool:
        """Cache an LLM response by query hash."""
        return self.set(f"cache:{query_hash}", response, ex=ttl)

    def get_cached_response(self, query_hash: str) -> Optional[Any]:
        """Retrieve a cached response."""
        return self.get(f"cache:{query_hash}")

    def store_session(self, session_id: str, user_data: Dict, ttl: int = 86400) -> bool:
        """Store session data."""
        return self.set(f"session:{session_id}", user_data, ex=ttl)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data."""
        return self.get(f"session:{session_id}")

    def sliding_window_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int]:
        """
        Atomic sliding window rate limit check using Lua script.

        Returns:
            Tuple of (allowed, current_count, retry_after_seconds)
        """
        if not self._ensure_connected():
            return True, 0, 0  # Fail open

        script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local member = ARGV[4]

        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
        local count = redis.call('ZCARD', key)

        if count < limit then
            redis.call('ZADD', key, now, member)
            redis.call('EXPIRE', key, window + 1)
            return {1, count + 1, 0}
        else
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = window
            if #oldest > 0 then
                retry_after = math.ceil((tonumber(oldest[2]) + window) - now)
            end
            return {0, count, retry_after}
        end
        """

        now = time.time()
        member = f"{now}:{id(self)}"

        try:
            result = self._client.eval(script, 1, key, now, window_seconds, limit, member)
            allowed = bool(result[0])
            current = int(result[1])
            retry_after = int(result[2])
            return allowed, current, retry_after
        except RedisError as exc:
            logger.error("Sliding window rate limit failed: %s", exc)
            return True, 0, 0

    def health_check(self) -> Dict[str, Any]:
        """Detailed health check for monitoring."""
        try:
            start = time.time()
            self._ensure_connected()
            if self._client:
                self._client.ping()
                latency_ms = round((time.time() - start) * 1000, 2)
                return {"healthy": True, "latency_ms": latency_ms}
            return {"healthy": False, "error": "Not connected"}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}


# ===========================================================================
# Async Redis Client (for asyncio code)
# ===========================================================================

class AsyncRedisClient:
    """
    Asynchronous Redis client wrapper.

    Used in async contexts (WebSocket handlers, async agent loops).
    """

    def __init__(self):
        self.client: Optional[async_redis.Redis] = None
        self._connecting_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish async connection with locking to prevent race conditions."""
        async with self._connecting_lock:
            if self.client:
                try:
                    await self.client.ping()
                    return
                except Exception:
                    self.client = None

            try:
                self.client = async_redis.from_url(
                    settings.redis_url,
                    password=settings.redis_password.get_secret_value() if settings.redis_password else None,
                    decode_responses=True,
                    socket_timeout=settings.redis_socket_timeout,
                    socket_connect_timeout=settings.redis_socket_timeout,
                    retry_on_timeout=settings.redis_retry_on_timeout,
                    max_connections=settings.redis_max_connections,
                )
                await self.client.ping()
                logger.info("Redis connected (async)")
            except Exception as exc:
                logger.error("Async Redis connection failed: %s", exc)
                self.client = None

    async def ensure_connected(self) -> bool:
        """Ensure connection is alive, reconnect if needed."""
        if self.client is None:
            await self.connect()
        else:
            try:
                await self.client.ping()
            except Exception:
                logger.warning("Async Redis ping failed, reconnecting...")
                await self.connect()
        return self.client is not None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set a key asynchronously."""
        if not await self.ensure_connected():
            return False
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            if ttl_seconds:
                await self.client.setex(key, ttl_seconds, value)
            else:
                await self.client.set(key, value)
            return True
        except RedisError as exc:
            logger.error("Async Redis SET failed: %s", exc)
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get a value asynchronously."""
        if not await self.ensure_connected():
            return None
        try:
            value = await self.client.get(key)
            if value is None:
                return None
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except RedisError as exc:
            logger.error("Async Redis GET failed: %s", exc)
            return None

    async def delete(self, key: str) -> bool:
        """Delete a key asynchronously."""
        if not await self.ensure_connected():
            return False
        try:
            await self.client.delete(key)
            return True
        except RedisError:
            return False

    async def exists(self, key: str) -> bool:
        """Check key existence asynchronously."""
        if not await self.ensure_connected():
            return False
        try:
            return await self.client.exists(key) > 0
        except RedisError:
            return False

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiry asynchronously."""
        if not await self.ensure_connected():
            return False
        try:
            return await self.client.expire(key, ttl_seconds)
        except RedisError:
            return False

    async def cache_response(self, query_hash: str, response: Any, ttl: int = 3600) -> bool:
        """Cache a response asynchronously."""
        return await self.set(f"cache:{query_hash}", response, ttl)

    async def get_cached_response(self, query_hash: str) -> Optional[Any]:
        """Get cached response asynchronously."""
        return await self.get(f"cache:{query_hash}")

    async def store_session(self, session_id: str, user_data: Dict, ttl: int = 86400) -> bool:
        """Store session asynchronously."""
        return await self.set(f"session:{session_id}", user_data, ttl)

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session asynchronously."""
        return await self.get(f"session:{session_id}")

    async def health_check(self) -> bool:
        """Async health check."""
        if not await self.ensure_connected():
            return False
        try:
            await self.client.ping()
            return True
        except Exception:
            return False

    async def sliding_window_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int, int]:
        """
        Async sliding window rate limit.

        Returns:
            Tuple of (allowed, current_count, retry_after_seconds)
        """
        if not await self.ensure_connected():
            return True, 0, 0

        now = time.time()

        try:
            await self.client.zremrangebyscore(key, 0, now - window_seconds)
            current = await self.client.zcard(key)

            if current >= limit:
                oldest = await self.client.zrange(key, 0, 0, withscores=True)
                retry_after = (
                    int((oldest[0][1] + window_seconds) - now) + 1
                    if oldest else window_seconds
                )
                return False, current, retry_after

            await self.client.zadd(key, {str(now): now})
            await self.client.expire(key, window_seconds)
            return True, current + 1, 0

        except RedisError as exc:
            logger.error("Async sliding window rate limit failed: %s", exc)
            return True, 0, 0

    async def close(self) -> None:
        """Close the connection gracefully."""
        if self.client:
            await self.client.close()
            self.client = None


# ===========================================================================
# Singletons
# ===========================================================================
_sync_client: Optional[SyncRedisClient] = None
_async_client: Optional[AsyncRedisClient] = None


def get_redis_client() -> SyncRedisClient:
    """Get or create the global sync Redis client singleton."""
    global _sync_client
    if _sync_client is None:
        _sync_client = SyncRedisClient()
    return _sync_client


async def get_async_redis_client() -> AsyncRedisClient:
    """Get or create the global async Redis client singleton."""
    global _async_client
    if _async_client is None:
        _async_client = AsyncRedisClient()
        await _async_client.connect()
    return _async_client


# Backward compatible singletons
redis_client = get_redis_client()
async_redis_client = AsyncRedisClient()