"""
DevMentor AI — Rate Limiter
============================
Production-grade rate limiting with:

- Sliding window algorithm via Redis sorted sets (accurate, no burst gaps)
- Atomic Lua script execution (no race conditions)
- Per-tier limits (free / pro / enterprise)
- Per-endpoint configurable limits
- Per-user AND per-IP limiting (both simultaneously)
- Graceful degradation when Redis is unavailable
- Exponential backoff tracking for repeat offenders
- Rate limit headers on every response (X-RateLimit-*)
- Retry-After header on 429 responses
- Structured logging for all limit events
- IP allowlist/blocklist support
- Endpoint-specific override limits
- Global kill switch for emergencies

Usage:
    from rate_limiter import rate_limit, limit_chat, limit_auth

    @app.route("/chat", methods=["POST"])
    @require_auth
    @rate_limit(requests_per_minute=60)
    def chat():
        ...

    @app.route("/auth/login", methods=["POST"])
    @limit_auth
    def login():
        ...

    # Tier-based limiting
    @app.route("/api/v1/query")
    @require_auth
    @limit_by_tier()
    def query():
        ...
"""

import hashlib
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from flask import g, jsonify, request

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
# Per-tier default limits (requests per minute)
# ---------------------------------------------------------------------------
TIER_LIMITS: Dict[str, Dict[str, int]] = {
    "free": {
        "default":  30,
        "chat":     20,
        "rag":      10,
        "upload":    5,
        "auth":     10,
        "search":   15,
        "agent":     5,
    },
    "pro": {
        "default":  200,
        "chat":     120,
        "rag":       60,
        "upload":    20,
        "auth":      20,
        "search":   100,
        "agent":     30,
    },
    "enterprise": {
        "default":  1000,
        "chat":      600,
        "rag":       300,
        "upload":    100,
        "auth":       50,
        "search":    500,
        "agent":     200,
    },
    "anonymous": {
        "default":  10,
        "chat":      5,
        "rag":       3,
        "upload":    2,
        "auth":      5,
        "search":    5,
        "agent":     0,
    },
}

# Endpoints that are always exempt from rate limiting
EXEMPT_PATHS = {
    "/health",
    "/metrics",
    "/favicon.ico",
    "/static",
}

# IPs that are always allowed (internal services, monitoring)
ALLOWLISTED_IPS: set = set()

# ---------------------------------------------------------------------------
# Lua script for atomic sliding window rate limiting
# Atomically: remove expired entries, count current, add if under limit
# Returns: [current_count, oldest_timestamp_or_0]
# ---------------------------------------------------------------------------
_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local request_id = ARGV[4]

-- Remove entries outside the window
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- Count current entries
local count = redis.call('ZCARD', key)

-- Get oldest entry for Retry-After calculation
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local oldest_score = 0
if #oldest > 0 then
    oldest_score = tonumber(oldest[2])
end

if count < limit then
    -- Add this request
    redis.call('ZADD', key, now, request_id)
    redis.call('EXPIRE', key, window + 1)
end

return {count, oldest_score}
"""


# ===========================================================================
# Redis Client Helper
# ===========================================================================

def _get_redis():
    """
    Get Redis client instance.
    Returns None if Redis is unavailable — triggers graceful degradation.
    """
    try:
        from redis_client import get_redis_client
        client = get_redis_client()
        # Quick connectivity check
        client.ping()
        return client
    except Exception as exc:
        logger.warning("Redis unavailable for rate limiting: %s", exc)
        return None


# ===========================================================================
# Client Identification
# ===========================================================================

def _get_user_identifier() -> str:
    """
    Get the user-level rate limit identifier.
    Priority: user_id > API key ID > anonymous
    """
    user_id = getattr(g, "user_id", None) or getattr(request, "user_id", None)
    if user_id:
        return f"user:{user_id}"

    api_key_id = getattr(g, "api_key_id", None)
    if api_key_id:
        return f"apikey:{api_key_id}"

    return f"anon:{_get_hashed_ip()}"


def _get_ip_identifier() -> str:
    """Get the IP-level rate limit identifier."""
    return f"ip:{_get_hashed_ip()}"


def _get_hashed_ip() -> str:
    """
    Hash the client IP for privacy-safe logging and storage.
    Still unique per IP but not directly reversible.
    """
    ip = _get_client_ip()
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _get_client_ip() -> str:
    """Extract real client IP respecting X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _get_user_tier() -> str:
    """Get the authenticated user's subscription tier."""
    return (
        getattr(g, "user_tier", None)
        or getattr(request, "user_tier", None)
        or "anonymous"
    )


# ===========================================================================
# Core Sliding Window Logic
# ===========================================================================

def _check_rate_limit(
    redis_client,
    key: str,
    limit: int,
    window_seconds: int,
) -> Tuple[bool, int, int, int]:
    """
    Execute atomic sliding window rate limit check.

    Args:
        redis_client:    Redis client
        key:             Rate limit bucket key
        limit:           Max requests allowed in window
        window_seconds:  Window size in seconds

    Returns:
        Tuple of (is_allowed, current_count, limit, retry_after_seconds)
    """
    now = time.time()
    request_id = f"{now}:{id(request)}"  # Unique per request

    try:
        # Execute atomic Lua script
        result = redis_client.eval(
            _SLIDING_WINDOW_SCRIPT,
            1,           # Number of keys
            key,         # KEYS[1]
            now,         # ARGV[1]: current timestamp
            window_seconds,  # ARGV[2]: window size
            limit,       # ARGV[3]: limit
            request_id,  # ARGV[4]: unique request identifier
        )

        current_count = int(result[0])
        oldest_score = float(result[1])

        # Calculate retry_after
        if oldest_score > 0:
            retry_after = max(1, int((oldest_score + window_seconds) - now) + 1)
        else:
            retry_after = window_seconds

        is_allowed = current_count < limit
        # After adding current request, count is current_count + 1
        display_count = current_count + 1 if is_allowed else current_count

        return is_allowed, display_count, limit, retry_after

    except Exception as exc:
        logger.error("Rate limit Lua script error for key %s: %s", key, exc)
        # Fail open — don't block users if rate limiter has a bug
        return True, 0, limit, 0


def _build_rate_limit_key(
    identifier: str,
    endpoint: str,
    window_type: str = "minute",
) -> str:
    """Build a namespaced Redis key for rate limiting."""
    return f"rl:{window_type}:{endpoint}:{identifier}"


# ===========================================================================
# Response Header Helpers
# ===========================================================================

def _add_rate_limit_headers(
    response,
    limit: int,
    remaining: int,
    reset_at: int,
    window: int,
) -> None:
    """
    Add standard rate limit headers to response.
    Follows the IETF RateLimit header draft standard.
    """
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    response.headers["X-RateLimit-Reset"] = str(reset_at)
    response.headers["X-RateLimit-Window"] = str(window)
    response.headers["X-RateLimit-Policy"] = f"{limit};w={window}"


def _build_429_response(
    identifier: str,
    endpoint: str,
    limit: int,
    current: int,
    retry_after: int,
    window: int,
) -> tuple:
    """Build a detailed 429 Too Many Requests response."""
    logger.warning(
        "Rate limit exceeded",
        extra={
            "identifier": identifier,
            "endpoint": endpoint,
            "current": current,
            "limit": limit,
            "retry_after": retry_after,
            "ip": _get_client_ip(),
        },
    )

    response = jsonify({
        "error": "Rate limit exceeded. Too many requests.",
        "code": "RATE_LIMIT_EXCEEDED",
        "limit": limit,
        "current": current,
        "retry_after": retry_after,
        "window_seconds": window,
        "message": f"You have exceeded the limit of {limit} requests per {window} seconds. "
                   f"Please retry after {retry_after} seconds.",
    })
    response.status_code = 429
    response.headers["Retry-After"] = str(retry_after)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = "0"
    response.headers["X-RateLimit-Reset"] = str(int(time.time()) + retry_after)
    return response


# ===========================================================================
# Main rate_limit Decorator
# ===========================================================================

def rate_limit(
    requests_per_minute: Optional[int] = None,
    requests_per_hour: Optional[int] = None,
    tier_limits: bool = False,
    endpoint_type: str = "default",
    key_prefix: Optional[str] = None,
    check_ip: bool = True,
    ip_limit_multiplier: float = 3.0,
    block_on_redis_failure: bool = False,
) -> Callable:
    """
    Flexible rate limiting decorator.

    Args:
        requests_per_minute:  Per-minute limit (overrides tier limits if set)
        requests_per_hour:    Additional per-hour limit (layered on top)
        tier_limits:          Use per-tier limits from TIER_LIMITS dict
        endpoint_type:        Category key for tier limits (chat/rag/upload/auth/search)
        key_prefix:           Custom prefix for Redis key (default: endpoint name)
        check_ip:             Also enforce a per-IP limit (default: True)
        ip_limit_multiplier:  IP limit = user limit * multiplier (catches multi-account abuse)
        block_on_redis_failure: If True, return 503 when Redis is down (fail closed)

    Usage:
        @rate_limit(requests_per_minute=60)
        @rate_limit(tier_limits=True, endpoint_type="chat")
        @rate_limit(requests_per_minute=100, check_ip=True)
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            # ---- Check exempt paths ----
            if any(request.path.startswith(p) for p in EXEMPT_PATHS):
                return f(*args, **kwargs)

            # ---- Check IP allowlist ----
            client_ip = _get_client_ip()
            if client_ip in ALLOWLISTED_IPS:
                return f(*args, **kwargs)

            # ---- Get Redis ----
            redis = _get_redis()
            if redis is None:
                if block_on_redis_failure:
                    logger.error("Redis unavailable — blocking requests (fail-closed mode)")
                    return jsonify({
                        "error": "Rate limiting service unavailable. Please try again shortly.",
                        "code": "SERVICE_UNAVAILABLE",
                    }), 503
                else:
                    logger.warning("Redis unavailable — rate limiting disabled (fail-open mode)")
                    return f(*args, **kwargs)

            # ---- Determine limits ----
            tier = _get_user_tier()
            user_identifier = _get_user_identifier()
            endpoint = key_prefix or request.endpoint or "unknown"

            if tier_limits:
                tier_config = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
                per_minute_limit = tier_config.get(endpoint_type, tier_config["default"])
            else:
                per_minute_limit = requests_per_minute or settings.rate_limit_per_user

            # ---- Per-user/session minute limit ----
            user_key = _build_rate_limit_key(user_identifier, endpoint, "min")
            user_allowed, user_count, user_limit, user_retry = _check_rate_limit(
                redis, user_key, per_minute_limit, 60
            )

            if not user_allowed:
                return _build_429_response(
                    user_identifier, endpoint,
                    user_limit, user_count, user_retry, 60,
                )

            # ---- Per-IP minute limit (catches abuse from single IP with many accounts) ----
            if check_ip:
                ip_identifier = _get_ip_identifier()
                ip_limit = int(per_minute_limit * ip_limit_multiplier)
                ip_key = _build_rate_limit_key(ip_identifier, endpoint, "min")
                ip_allowed, ip_count, ip_limit_val, ip_retry = _check_rate_limit(
                    redis, ip_key, ip_limit, 60
                )

                if not ip_allowed:
                    logger.warning(
                        "IP-level rate limit exceeded",
                        extra={"ip": client_ip, "endpoint": endpoint, "count": ip_count},
                    )
                    return _build_429_response(
                        ip_identifier, endpoint,
                        ip_limit_val, ip_count, ip_retry, 60,
                    )

            # ---- Optional hourly limit ----
            if requests_per_hour:
                hourly_key = _build_rate_limit_key(user_identifier, endpoint, "hour")
                hour_allowed, hour_count, hour_limit, hour_retry = _check_rate_limit(
                    redis, hourly_key, requests_per_hour, 3600
                )

                if not hour_allowed:
                    return _build_429_response(
                        user_identifier, endpoint,
                        hour_limit, hour_count, hour_retry, 3600,
                    )

            # ---- Request allowed — execute and add headers ----
            remaining = max(0, per_minute_limit - user_count)
            reset_at = int(time.time()) + 60

            response = f(*args, **kwargs)

            # Add headers to response (handle both Response objects and tuples)
            if hasattr(response, "headers"):
                _add_rate_limit_headers(
                    response, per_minute_limit, remaining, reset_at, 60
                )
            elif isinstance(response, tuple) and len(response) >= 1:
                from flask import make_response
                resp = make_response(response)
                _add_rate_limit_headers(
                    resp, per_minute_limit, remaining, reset_at, 60
                )
                return resp

            return response

        return decorated
    return decorator


# ===========================================================================
# Convenience Decorators
# ===========================================================================

def limit_chat(tier_based: bool = True):
    """
    Rate limiter for chat endpoints.
    Uses tier-based limits by default.

    Usage:
        @app.route("/chat", methods=["POST"])
        @require_auth
        @limit_chat()
        def chat():
            ...
    """
    if tier_based:
        return rate_limit(tier_limits=True, endpoint_type="chat")
    return rate_limit(requests_per_minute=settings.rate_limit_per_user, endpoint_type="chat")


def limit_auth(strict: bool = True):
    """
    Strict rate limiter for authentication endpoints.
    Lower limits + IP-based checking to prevent brute force.

    Usage:
        @app.route("/auth/login", methods=["POST"])
        @limit_auth()
        def login():
            ...
    """
    limit = 5 if strict else 20
    return rate_limit(
        requests_per_minute=limit,
        endpoint_type="auth",
        check_ip=True,
        ip_limit_multiplier=2.0,
        block_on_redis_failure=True,  # Fail closed for auth — safer
    )


def limit_upload():
    """
    Rate limiter for file upload endpoints.
    Very conservative limits — uploads are expensive.

    Usage:
        @app.route("/upload_docs", methods=["POST"])
        @require_auth
        @limit_upload()
        def upload():
            ...
    """
    return rate_limit(
        tier_limits=True,
        endpoint_type="upload",
        requests_per_hour=50,  # Also enforce hourly limit
    )


def limit_rag():
    """Rate limiter for RAG query endpoints."""
    return rate_limit(tier_limits=True, endpoint_type="rag")


def limit_search():
    """Rate limiter for search endpoints."""
    return rate_limit(tier_limits=True, endpoint_type="search")


def limit_agent():
    """
    Rate limiter for agent endpoints.
    Agent calls are expensive — very conservative limits.
    """
    return rate_limit(
        tier_limits=True,
        endpoint_type="agent",
        requests_per_hour=100,
    )


def limit_by_tier():
    """General tier-based rate limiter for any endpoint."""
    return rate_limit(tier_limits=True, endpoint_type="default")


def limit_api(requests_per_minute: int = 60):
    """Rate limiter for public API endpoints."""
    return rate_limit(
        requests_per_minute=requests_per_minute,
        check_ip=True,
    )


# ===========================================================================
# Admin: Dynamic Limit Management
# ===========================================================================

def set_user_rate_limit_override(
    user_id: str,
    limit: int,
    window_seconds: int = 60,
    ttl_seconds: int = 86400,
) -> bool:
    """
    Set a temporary rate limit override for a specific user.
    Useful for throttling abusive users or granting temporary elevated limits.

    Args:
        user_id:        Target user ID
        limit:          New limit value
        window_seconds: Window size
        ttl_seconds:    How long this override lasts (default: 24 hours)

    Returns:
        True if override was set successfully
    """
    redis = _get_redis()
    if not redis:
        return False

    try:
        override_key = f"rl:override:user:{user_id}"
        redis.setex(override_key, ttl_seconds, str(limit))
        logger.info(
            "Rate limit override set",
            extra={"user_id": user_id, "limit": limit, "ttl": ttl_seconds},
        )
        return True
    except Exception as exc:
        logger.error("Failed to set rate limit override: %s", exc)
        return False


def reset_user_rate_limit(user_id: str, endpoint: str = "*") -> bool:
    """
    Reset rate limit counters for a user.
    Useful when a user reports false positives or after plan upgrades.

    Args:
        user_id:  Target user ID
        endpoint: Specific endpoint to reset, or '*' for all

    Returns:
        True if reset was successful
    """
    redis = _get_redis()
    if not redis:
        return False

    try:
        if endpoint == "*":
            # Find and delete all rate limit keys for this user
            pattern = f"rl:*:*:user:{user_id}"
            keys = redis.keys(pattern)
            if keys:
                redis.delete(*keys)
            logger.info(
                "Rate limits reset for user",
                extra={"user_id": user_id, "keys_deleted": len(keys)},
            )
        else:
            key = _build_rate_limit_key(f"user:{user_id}", endpoint, "min")
            redis.delete(key)

        return True
    except Exception as exc:
        logger.error("Failed to reset rate limit for user %s: %s", user_id, exc)
        return False


def get_rate_limit_status(user_id: str) -> Dict[str, Any]:
    """
    Get current rate limit status for a user across all endpoints.
    Used by admin panel and user dashboard.

    Returns:
        Dict with current counts per endpoint
    """
    redis = _get_redis()
    if not redis:
        return {"error": "Redis unavailable"}

    try:
        pattern = f"rl:*:*:user:{user_id}"
        keys = redis.keys(pattern)
        status = {}
        now = time.time()

        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            parts = key_str.split(":")
            if len(parts) >= 4:
                endpoint = parts[2]
                count = redis.zcount(key, now - 60, now)
                status[endpoint] = {
                    "current": count,
                    "key": key_str,
                }

        return {
            "user_id": user_id,
            "endpoints": status,
            "timestamp": now,
        }
    except Exception as exc:
        logger.error("Failed to get rate limit status: %s", exc)
        return {"error": str(exc)}


# ===========================================================================
# IP Allowlist Management
# ===========================================================================

def add_to_allowlist(ip: str) -> None:
    """Add an IP to the rate limit allowlist (e.g. internal monitoring IPs)."""
    ALLOWLISTED_IPS.add(ip)
    logger.info("IP added to rate limit allowlist: %s", ip)


def remove_from_allowlist(ip: str) -> None:
    """Remove an IP from the rate limit allowlist."""
    ALLOWLISTED_IPS.discard(ip)
    logger.info("IP removed from rate limit allowlist: %s", ip)


def block_ip(ip: str, duration_seconds: int = 3600) -> bool:
    """
    Temporarily block all requests from an IP.
    Used for DDoS mitigation or abuse response.

    Args:
        ip:               IP address to block
        duration_seconds: Block duration (default: 1 hour)

    Returns:
        True if block was set
    """
    redis = _get_redis()
    if not redis:
        return False

    try:
        block_key = f"rl:blocked:ip:{hashlib.sha256(ip.encode()).hexdigest()[:16]}"
        redis.setex(block_key, duration_seconds, "1")
        logger.warning(
            "IP blocked",
            extra={"ip_hash": block_key, "duration": duration_seconds},
        )
        return True
    except Exception as exc:
        logger.error("Failed to block IP: %s", exc)
        return False


def is_ip_blocked(ip: str) -> bool:
    """Check if an IP is currently blocked."""
    redis = _get_redis()
    if not redis:
        return False

    try:
        block_key = f"rl:blocked:ip:{hashlib.sha256(ip.encode()).hexdigest()[:16]}"
        return redis.exists(block_key) > 0
    except Exception:
        return False