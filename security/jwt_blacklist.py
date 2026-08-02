"""
DevMentor AI — JWT Blacklist
=============================
Production-grade JWT token blacklisting with:

- Redis-backed blacklist with automatic TTL expiry
- Connection pooling with retry logic
- Bulk revocation (revoke all tokens for a user)
- Blacklist statistics for monitoring
- Graceful degradation on Redis failure
- Structured logging for all revocation events
- Thread-safe operations
- Key namespacing to avoid collisions

How it works:
    When a user logs out or changes password, their token's JTI
    (JWT ID) is stored in Redis with a TTL matching the token's
    remaining lifetime. On every authenticated request, the JTI
    is checked against this blacklist before the request proceeds.

    Since Redis automatically expires keys, the blacklist stays
    lean — no manual cleanup needed.

Usage:
    from security.jwt_blacklist import add_to_blacklist, is_blacklisted

    # On logout:
    add_to_blacklist(jti, expires_in_seconds=3600)

    # On every auth check:
    if is_blacklisted(jti):
        return 401

    # Revoke all tokens for a user (e.g. password change):
    revoke_all_user_tokens(user_id)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

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
# Key Prefixes — namespaced to avoid collisions with other Redis data
# ---------------------------------------------------------------------------
_JTI_PREFIX = "jwtbl:jti:"          # Per-token blacklist entry
_USER_PREFIX = "jwtbl:user:"        # Per-user token set (for bulk revocation)
_STATS_KEY = "jwtbl:stats"          # Hash of blacklist statistics
_REVOKE_ALL_KEY = "jwtbl:revoke_all:"  # Per-user "revoke all before" timestamp


# ===========================================================================
# Redis Connection
# ===========================================================================

def _build_redis_client() -> redis.Redis:
    """
    Build a Redis client with:
    - Connection pooling
    - Automatic retry with exponential backoff
    - Socket timeout to prevent hanging
    - Health check on connection
    """
    retry = Retry(
        ExponentialBackoff(cap=10, base=1),
        retries=3,
    )

    client = redis.from_url(
        str(settings.redis_url),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        socket_keepalive=True,
        retry=retry,
        retry_on_error=[ConnectionError, TimeoutError],
        health_check_interval=30,
        max_connections=settings.redis_max_connections,
    )

    return client


# Singleton Redis client for this module
_redis_client: Optional[redis.Redis] = None


def _get_client() -> Optional[redis.Redis]:
    """
    Get the Redis client, initializing it on first call.
    Returns None if Redis is unavailable.
    """
    global _redis_client

    if _redis_client is None:
        try:
            _redis_client = _build_redis_client()
            _redis_client.ping()
            logger.info("JWT blacklist Redis connection established")
        except Exception as exc:
            logger.error("JWT blacklist Redis connection failed: %s", exc)
            _redis_client = None

    return _redis_client


def _reset_client() -> None:
    """Reset the Redis client — called after connection failures."""
    global _redis_client
    _redis_client = None


# ===========================================================================
# Core Blacklist Operations
# ===========================================================================

def add_to_blacklist(
    jti: str,
    expires_in_seconds: int,
    user_id: Optional[str] = None,
    reason: str = "logout",
) -> bool:
    """
    Add a JWT JTI to the blacklist.

    The entry automatically expires when the token would have expired anyway,
    so no manual cleanup is needed.

    Args:
        jti:                 JWT ID claim from the token payload
        expires_in_seconds:  Seconds until the token expires (sets Redis TTL)
        user_id:             Optional user ID for per-user tracking
        reason:              Why the token was revoked (logout/password_change/admin)

    Returns:
        True if successfully blacklisted, False on Redis failure

    Example:
        # On user logout:
        payload = decode_access_token(token)
        remaining = payload['exp'] - int(time.time())
        add_to_blacklist(payload['jti'], remaining, user_id=user.id, reason='logout')
    """
    if not jti:
        logger.warning("add_to_blacklist called with empty JTI")
        return False

    # Ensure TTL is positive — don't store already-expired tokens
    if expires_in_seconds <= 0:
        logger.debug("Token already expired, skipping blacklist for JTI %s", jti)
        return True  # Already expired = effectively blacklisted

    client = _get_client()
    if client is None:
        logger.error(
            "Cannot blacklist JTI %s — Redis unavailable. "
            "Token remains valid until natural expiry!",
            jti,
        )
        return False

    try:
        key = f"{_JTI_PREFIX}{jti}"

        # Pipeline for atomicity — both ops succeed or neither does
        pipe = client.pipeline(transaction=True)

        # Store blacklist entry with TTL
        pipe.setex(
            key,
            expires_in_seconds + 60,  # +60s buffer for clock skew
            reason,
        )

        # Track per-user blacklisted tokens (for bulk revocation queries)
        if user_id:
            user_key = f"{_USER_PREFIX}{user_id}"
            pipe.sadd(user_key, jti)
            # User token set expires after the longest possible token lifetime
            pipe.expire(user_key, max(expires_in_seconds + 60, 86400 * 30))

        # Increment stats
        pipe.hincrby(_STATS_KEY, "total_revoked", 1)
        pipe.hset(_STATS_KEY, "last_revoked_at", int(time.time()))

        pipe.execute()

        logger.info(
            "Token blacklisted",
            extra={
                "jti": jti,
                "user_id": user_id,
                "expires_in": expires_in_seconds,
                "reason": reason,
            },
        )
        return True

    except RedisError as exc:
        logger.error("Redis error blacklisting JTI %s: %s", jti, exc)
        _reset_client()
        return False
    except Exception as exc:
        logger.error("Unexpected error blacklisting JTI %s: %s", jti, exc)
        return False


def is_blacklisted(jti: str) -> bool:
    """
    Check if a JWT JTI is in the blacklist.

    This is called on EVERY authenticated request, so it must be fast.
    Redis GET is O(1) — typically sub-millisecond.

    Args:
        jti: JWT ID claim from the token payload

    Returns:
        True if token is blacklisted (should be rejected)
        False if token is valid OR if Redis is unavailable (fail open)

    Security note:
        On Redis failure, we fail OPEN (allow the request) rather than
        fail closed (block all requests). This is the standard trade-off
        for availability vs security. If you need fail-closed behavior,
        change the except clause to return True.
    """
    if not jti:
        return False

    client = _get_client()
    if client is None:
        logger.warning(
            "Blacklist check skipped — Redis unavailable. "
            "Failing open for JTI: %s",
            jti,
        )
        # FAIL OPEN: return False (allow request)
        # Change to: return True  for FAIL CLOSED behavior
        return False

    try:
        key = f"{_JTI_PREFIX}{jti}"
        result = client.exists(key)
        blacklisted = result > 0

        if blacklisted:
            logger.warning(
                "Blacklisted token presented",
                extra={"jti": jti},
            )

        return blacklisted

    except RedisError as exc:
        logger.error("Redis error checking blacklist for JTI %s: %s", jti, exc)
        _reset_client()
        return False  # Fail open
    except Exception as exc:
        logger.error("Unexpected error checking blacklist for JTI %s: %s", jti, exc)
        return False


def get_revocation_reason(jti: str) -> Optional[str]:
    """
    Get the reason a token was revoked.
    Useful for audit logs and debugging.

    Args:
        jti: JWT ID to look up

    Returns:
        Revocation reason string, or None if not blacklisted
    """
    client = _get_client()
    if client is None:
        return None

    try:
        return client.get(f"{_JTI_PREFIX}{jti}")
    except RedisError as exc:
        logger.error("Redis error getting revocation reason: %s", exc)
        return None


# ===========================================================================
# Bulk Revocation
# ===========================================================================

def revoke_all_user_tokens(
    user_id: str,
    reason: str = "password_change",
) -> bool:
    """
    Revoke ALL active tokens for a user.

    Uses a "revoke all before timestamp" approach — more efficient than
    tracking individual JTIs for bulk operations.

    When a user changes their password or is banned, all their existing
    tokens become invalid regardless of their individual JTIs.

    Args:
        user_id: User whose tokens should all be revoked
        reason:  Why tokens are being revoked

    Returns:
        True if revocation timestamp was set successfully

    How it works:
        Sets a "revoke_all_before" timestamp for the user.
        The auth middleware checks this timestamp against the token's
        "iat" (issued at) claim — if iat < revoke_all_before, token is rejected.
    """
    client = _get_client()
    if client is None:
        logger.error("Cannot revoke all tokens for user %s — Redis unavailable", user_id)
        return False

    try:
        revoke_key = f"{_REVOKE_ALL_KEY}{user_id}"
        now = int(time.time())

        # Store timestamp — all tokens issued BEFORE this time are invalid
        # TTL = 30 days (maximum token lifetime)
        client.setex(revoke_key, 86400 * 30, str(now))

        # Increment stats
        client.hincrby(_STATS_KEY, "bulk_revocations", 1)

        logger.info(
            "All tokens revoked for user",
            extra={
                "user_id": user_id,
                "revoked_before": now,
                "reason": reason,
            },
        )
        return True

    except RedisError as exc:
        logger.error(
            "Redis error revoking all tokens for user %s: %s",
            user_id, exc,
        )
        _reset_client()
        return False


def is_user_globally_revoked(user_id: str, token_issued_at: int) -> bool:
    """
    Check if a token was issued before the user's global revocation timestamp.

    Call this in addition to is_blacklisted() for complete revocation coverage.

    Args:
        user_id:          User ID from token
        token_issued_at:  "iat" claim from JWT payload (Unix timestamp)

    Returns:
        True if token predates the revocation (should be rejected)

    Usage in auth_middleware:
        payload = decode_access_token(token)
        if is_user_globally_revoked(payload['sub'], payload['iat']):
            return 401
    """
    client = _get_client()
    if client is None:
        return False  # Fail open

    try:
        revoke_key = f"{_REVOKE_ALL_KEY}{user_id}"
        revoke_before = client.get(revoke_key)

        if revoke_before is None:
            return False  # No global revocation set

        revoke_timestamp = int(revoke_before)
        is_revoked = token_issued_at < revoke_timestamp

        if is_revoked:
            logger.warning(
                "Globally revoked token presented",
                extra={
                    "user_id": user_id,
                    "token_iat": token_issued_at,
                    "revoke_before": revoke_timestamp,
                },
            )

        return is_revoked

    except (RedisError, ValueError) as exc:
        logger.error("Error checking global revocation for user %s: %s", user_id, exc)
        return False


def revoke_tokens_list(
    jtis: List[str],
    expires_in_seconds: int,
    user_id: Optional[str] = None,
    reason: str = "bulk_revoke",
) -> int:
    """
    Revoke multiple tokens in a single Redis pipeline call.
    Much more efficient than calling add_to_blacklist() in a loop.

    Args:
        jtis:                List of JTI strings to revoke
        expires_in_seconds:  TTL for each blacklist entry
        user_id:             Optional user ID for tracking
        reason:              Revocation reason

    Returns:
        Number of tokens successfully blacklisted
    """
    if not jtis:
        return 0

    client = _get_client()
    if client is None:
        logger.error("Cannot bulk revoke %d tokens — Redis unavailable", len(jtis))
        return 0

    try:
        pipe = client.pipeline(transaction=False)  # Non-transactional for performance

        for jti in jtis:
            key = f"{_JTI_PREFIX}{jti}"
            pipe.setex(key, expires_in_seconds + 60, reason)

        if user_id:
            user_key = f"{_USER_PREFIX}{user_id}"
            pipe.sadd(user_key, *jtis)
            pipe.expire(user_key, 86400 * 30)

        pipe.hincrby(_STATS_KEY, "total_revoked", len(jtis))
        pipe.execute()

        logger.info(
            "Bulk token revocation complete",
            extra={
                "count": len(jtis),
                "user_id": user_id,
                "reason": reason,
            },
        )
        return len(jtis)

    except RedisError as exc:
        logger.error("Redis error during bulk revocation: %s", exc)
        _reset_client()
        return 0


# ===========================================================================
# Cleanup & Maintenance
# ===========================================================================

def cleanup_expired_user_sets(user_id: str) -> int:
    """
    Remove expired JTIs from a user's token tracking set.
    Redis auto-expires individual JTI keys, but the user set entry
    may still reference them. This cleans up stale references.

    Args:
        user_id: User whose tracking set to clean

    Returns:
        Number of stale JTIs removed
    """
    client = _get_client()
    if client is None:
        return 0

    try:
        user_key = f"{_USER_PREFIX}{user_id}"
        jtis = client.smembers(user_key)

        if not jtis:
            return 0

        stale = []
        pipe = client.pipeline(transaction=False)

        for jti in jtis:
            pipe.exists(f"{_JTI_PREFIX}{jti}")

        results = pipe.execute()

        pipe = client.pipeline(transaction=False)
        for jti, exists in zip(jtis, results):
            if not exists:
                stale.append(jti)
                pipe.srem(user_key, jti)

        if stale:
            pipe.execute()
            logger.debug(
                "Cleaned %d stale JTI references for user %s",
                len(stale), user_id,
            )

        return len(stale)

    except RedisError as exc:
        logger.error("Error cleaning user token set for %s: %s", user_id, exc)
        return 0


# ===========================================================================
# Statistics & Health
# ===========================================================================

def get_blacklist_stats() -> Dict[str, Any]:
    """
    Get blacklist statistics for monitoring and admin dashboard.

    Returns:
        Dict with total_revoked, bulk_revocations, last_revoked_at,
        redis_connected, estimated_active_blacklist_size
    """
    client = _get_client()

    if client is None:
        return {
            "redis_connected": False,
            "total_revoked": 0,
            "bulk_revocations": 0,
            "last_revoked_at": None,
            "estimated_active_blacklist_size": 0,
            "error": "Redis unavailable",
        }

    try:
        stats = client.hgetall(_STATS_KEY)

        # Count active blacklist entries (approximate via SCAN)
        active_count = 0
        cursor = 0
        while True:
            cursor, keys = client.scan(
                cursor=cursor,
                match=f"{_JTI_PREFIX}*",
                count=100,
            )
            active_count += len(keys)
            if cursor == 0:
                break

        last_revoked = stats.get("last_revoked_at")
        last_revoked_dt = (
            datetime.fromtimestamp(int(last_revoked), tz=timezone.utc).isoformat()
            if last_revoked
            else None
        )

        return {
            "redis_connected": True,
            "total_revoked": int(stats.get("total_revoked", 0)),
            "bulk_revocations": int(stats.get("bulk_revocations", 0)),
            "last_revoked_at": last_revoked_dt,
            "estimated_active_blacklist_size": active_count,
        }

    except RedisError as exc:
        logger.error("Error getting blacklist stats: %s", exc)
        _reset_client()
        return {"redis_connected": False, "error": str(exc)}


def check_blacklist_health() -> Dict[str, Any]:
    """
    Health check for the JWT blacklist system.
    Used by /health endpoint.

    Returns:
        Dict with status (healthy/degraded/unhealthy) and details
    """
    client = _get_client()

    if client is None:
        return {
            "status": "unhealthy",
            "message": "Redis connection failed — blacklisting disabled",
            "impact": "Revoked tokens may be accepted until Redis recovers",
        }

    try:
        # Write and read a test key
        test_key = "jwtbl:healthcheck"
        client.setex(test_key, 10, "ok")
        result = client.get(test_key)

        if result == "ok":
            return {
                "status": "healthy",
                "message": "JWT blacklist operational",
            }
        else:
            return {
                "status": "degraded",
                "message": "Redis read/write mismatch",
            }

    except RedisError as exc:
        _reset_client()
        return {
            "status": "unhealthy",
            "message": f"Redis error: {exc}",
            "impact": "Revoked tokens may be accepted",
        }


# ===========================================================================
# Backward Compatibility Aliases
# ===========================================================================
# These match the function names used in auth_middleware.py

def revoke_token(jti: str, expires_in: int, user_id: Optional[str] = None) -> bool:
    """Alias for add_to_blacklist() — backward compatibility."""
    return add_to_blacklist(jti, expires_in, user_id=user_id, reason="revoked")


def is_revoked(jti: str) -> bool:
    """Alias for is_blacklisted() — backward compatibility."""
    return is_blacklisted(jti)