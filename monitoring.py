"""
DevMentor AI — Monitoring Service
====================================
Production-grade monitoring with:

- Prometheus metrics with low cardinality grouping
- Custom metrics: chat requests, token usage, RAG hits, cache rates
- Kubernetes liveness + readiness + detailed health probes
- Parallel health checks with timeouts
- System resource monitoring (CPU, memory, disk)
- Slow request detection and logging
- Optional metrics endpoint authentication
- Request ID injection for distributed tracing
- Structured health check responses

UPGRADE NOTE (this revision — fixes a real boot-time error found running
the app for the first time):

_group_by_endpoint() was defined as a zero-argument function:
    def _group_by_endpoint():
        ...
but prometheus_flask_exporter calls any callable passed as `group_by`
with the current Flask `request` object as its one positional argument
(confirmed directly against the installed library's source — see
PrometheusMetrics.export_defaults, where it does `group = duration_group(request)`
whenever `callable(duration_group)` is True). Calling a zero-argument
function with one argument raises exactly the TypeError seen in the
traceback: "takes 0 positional arguments but 1 was given" — and because
this fires inside both Flask's after_request AND teardown_request hooks,
it fired on every single request, including /health, turning every health
check into a 500.

Fixed by accepting the request parameter — using Flask's own `request`
proxy inside the body is unaffected either way, but the function now
matches the call signature the library actually uses.

Usage:
    from monitoring import init_monitoring

    # In app factory
    monitoring = init_monitoring(app)
"""

import concurrent.futures
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional

import psutil
from flask import Flask, g, jsonify, request
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge, Histogram, Summary

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

# ---------------------------------------------------------------------------
# Slow request threshold
# ---------------------------------------------------------------------------
SLOW_REQUEST_THRESHOLD_MS = 2000


# ===========================================================================
# Custom Prometheus Metrics
# ===========================================================================

# Chat metrics
CHAT_REQUESTS_TOTAL = Counter(
    "devmentor_chat_requests_total",
    "Total chat requests",
    ["model", "provider", "status"],
)

CHAT_DURATION_SECONDS = Histogram(
    "devmentor_chat_duration_seconds",
    "Chat response duration in seconds",
    ["model", "provider"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Token metrics
TOKEN_USAGE_TOTAL = Counter(
    "devmentor_token_usage_total",
    "Total tokens consumed",
    ["model", "token_type"],
)

COST_USD_TOTAL = Counter(
    "devmentor_cost_usd_total",
    "Total API cost in USD",
    ["model", "provider"],
)

# RAG metrics
RAG_REQUESTS_TOTAL = Counter(
    "devmentor_rag_requests_total",
    "Total RAG query requests",
    ["status"],
)

RAG_CACHE_HITS_TOTAL = Counter(
    "devmentor_rag_cache_hits_total",
    "Total semantic cache hits",
    ["hit_type"],
)

# User metrics
ACTIVE_USERS_GAUGE = Gauge(
    "devmentor_active_users",
    "Currently active users (last 5 min)",
)

# Rate limiting metrics
RATE_LIMIT_EXCEEDED_TOTAL = Counter(
    "devmentor_rate_limit_exceeded_total",
    "Total rate limit rejections",
    ["endpoint", "tier"],
)

# Agent metrics
AGENT_RUNS_TOTAL = Counter(
    "devmentor_agent_runs_total",
    "Total agent execution runs",
    ["status"],
)

AGENT_ITERATIONS_HISTOGRAM = Histogram(
    "devmentor_agent_iterations",
    "Agent iterations per run",
    buckets=[1, 2, 3, 5, 8, 10, 15, 20],
)

# WebSocket metrics
WS_CONNECTIONS_GAUGE = Gauge(
    "devmentor_ws_connections_active",
    "Active WebSocket connections",
)

WS_MESSAGES_TOTAL = Counter(
    "devmentor_ws_messages_total",
    "Total WebSocket messages",
    ["event_type"],
)

# Error metrics
ERRORS_TOTAL = Counter(
    "devmentor_errors_total",
    "Total application errors",
    ["error_type", "endpoint"],
)

# Memory metrics
MEMORY_STORE_TOTAL = Counter(
    "devmentor_memory_store_total",
    "Total memories stored",
    ["memory_type"],
)


# ===========================================================================
# Helper: Run with Timeout
# ===========================================================================

def _run_with_timeout(fn: Callable, timeout_seconds: int = 5) -> Dict[str, Any]:
    """
    Run a function with a timeout.
    Returns error dict if timeout exceeded.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return {"healthy": False, "error": f"Health check timed out after {timeout_seconds}s"}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}


# ===========================================================================
# Monitoring Class
# ===========================================================================

class Monitoring:
    """
    Production-grade monitoring service for DevMentor AI.

    Provides:
    - Prometheus metrics via /metrics
    - Health probes via /health/*
    - Request middleware for timing and tracing
    """

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.prometheus = None
        if app:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        """
        Initialize monitoring with Flask app.
        Call from app factory.
        """
        self.app = app

        # ---- Prometheus Setup ----
        def _group_by_endpoint(req):
            """
            Low-cardinality endpoint grouping.
            Replaces dynamic IDs: /users/123 → /users/{id}

            FIX: prometheus_flask_exporter calls this with the current
            Flask request object as its one positional argument
            (`duration_group(request)` internally) — this function must
            accept that parameter even though Flask's `request` proxy is
            also globally available and would work identically here. The
            parameter name is `req` rather than shadowing the imported
            `request` proxy, to make the call signature mismatch
            unmistakable if anyone reads this function in isolation later.
            """
            if req.url_rule:
                return str(req.url_rule)
            return req.path

        self.prometheus = PrometheusMetrics(
            app,
            group_by=_group_by_endpoint,
            default_labels={
                "app": settings.app_name,
                "env": settings.app_env,
                "version": settings.app_version,
            },
            default_latency_metric=CHAT_DURATION_SECONDS,
        )

        self.prometheus.info(
            "app_info",
            "DevMentor AI application info",
            version=settings.app_version,
            environment=settings.app_env,
        )

        self._register_health_routes(app)
        self._register_middleware(app)

        logger.info(
            "Monitoring initialized: env=%s version=%s",
            settings.app_env, settings.app_version,
        )

    def _register_health_routes(self, app: Flask) -> None:
        """Register /health/* and /metrics endpoints."""

        @app.route("/health")
        @app.route("/health/liveness")
        def liveness():
            """
            Kubernetes liveness probe.
            Returns 200 if process is alive.
            """
            return jsonify({
                "status": "alive",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": settings.app_version,
                "env": settings.app_env,
            }), 200

        @app.route("/health/readiness")
        def readiness():
            """
            Kubernetes readiness probe.
            Returns 200 only if database is reachable.
            """
            db_status = _run_with_timeout(self._check_database, timeout_seconds=5)

            if db_status.get("healthy"):
                return jsonify({
                    "ready": True,
                    "database": "ok",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }), 200
            else:
                return jsonify({
                    "ready": False,
                    "database": "unavailable",
                    "error": db_status.get("error"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }), 503

        @app.route("/health/detailed")
        def detailed_health():
            """
            Comprehensive health check showing all dependencies.
            Runs all checks in parallel with timeouts.
            """
            # Optional: restrict to internal/admin access
            prometheus_key = os.getenv("PROMETHEUS_API_KEY")
            if prometheus_key:
                provided = request.headers.get("X-API-Key") or request.args.get("key")
                if provided != prometheus_key:
                    return jsonify({"error": "Unauthorized"}), 401

            start = time.time()

            # Run all health checks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    "database": executor.submit(_run_with_timeout, self._check_database, 5),
                    "redis":    executor.submit(_run_with_timeout, self._check_redis, 5),
                    "qdrant":   executor.submit(_run_with_timeout, self._check_qdrant, 5),
                    "celery":   executor.submit(_run_with_timeout, self._check_celery, 5),
                    "jwt_blacklist": executor.submit(_run_with_timeout, self._check_jwt_blacklist, 3),
                }
                results = {name: future.result() for name, future in futures.items()}

            system = self._get_system_info()

            # Overall health — database is required, others are optional
            is_healthy = results["database"].get("healthy", False)
            is_degraded = not all(
                v.get("healthy", False) for v in results.values()
            )

            overall = "healthy" if is_healthy and not is_degraded else (
                "degraded" if is_healthy else "unhealthy"
            )

            duration_ms = int((time.time() - start) * 1000)

            return jsonify({
                "status": overall,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": settings.app_version,
                "environment": settings.app_env,
                "check_duration_ms": duration_ms,
                "components": results,
                "system": system,
                "features": {
                    "streaming": settings.feature_streaming,
                    "rag": settings.feature_rag,
                    "agent": settings.feature_agent,
                    "memory": settings.feature_long_term_memory,
                },
            }), 200 if overall == "healthy" else 503

    def _register_middleware(self, app: Flask) -> None:
        """Register request timing and tracing middleware."""

        @app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        @app.after_request
        def after_request(response):
            # Inject request ID into response
            response.headers["X-Request-ID"] = getattr(g, "request_id", "")

            if hasattr(g, "start_time"):
                duration_ms = (time.time() - g.start_time) * 1000

                # Log slow requests
                if duration_ms > SLOW_REQUEST_THRESHOLD_MS:
                    logger.warning(
                        "Slow request: %s %s took %.0fms (status=%d)",
                        request.method,
                        request.path,
                        duration_ms,
                        response.status_code,
                    )

                # Track errors
                if response.status_code >= 500:
                    ERRORS_TOTAL.labels(
                        error_type="server_error",
                        endpoint=str(request.url_rule or request.path),
                    ).inc()

                elif response.status_code == 429:
                    tier = getattr(request, "user_tier", "unknown")
                    RATE_LIMIT_EXCEEDED_TOTAL.labels(
                        endpoint=str(request.url_rule or request.path),
                        tier=tier,
                    ).inc()

            return response

    # -----------------------------------------------------------------------
    # Health Check Implementations
    # -----------------------------------------------------------------------

    def _check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL connectivity."""
        try:
            from sqlalchemy import text
            with get_db() as db:
                result = db.execute(text("SELECT 1")).scalar()
                if result == 1:
                    return {"healthy": True, "latency_ms": None}
            return {"healthy": False, "error": "Unexpected query result"}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            from redis_client import get_redis_client
            redis = get_redis_client()
            start = time.time()
            redis.ping()
            latency_ms = round((time.time() - start) * 1000, 1)
            return {"healthy": True, "latency_ms": latency_ms}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    def _check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant vector DB connectivity."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=3,
            )
            start = time.time()
            client.get_collections()
            latency_ms = round((time.time() - start) * 1000, 1)
            return {"healthy": True, "latency_ms": latency_ms}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    def _check_celery(self) -> Dict[str, Any]:
        """Check Celery worker connectivity."""
        try:
            from celery_app import celery_app
            result = celery_app.control.ping(timeout=2)
            workers_online = len(result) if result else 0
            return {
                "healthy": workers_online > 0,
                "workers_online": workers_online,
            }
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    def _check_jwt_blacklist(self) -> Dict[str, Any]:
        """Check JWT blacklist (Redis) subsystem."""
        try:
            from security.jwt_blacklist import check_blacklist_health
            return check_blacklist_health()
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        info: Dict[str, Any] = {
            "pid": os.getpid(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        except Exception:
            info["cpu_percent"] = None

        try:
            mem = psutil.virtual_memory()
            info["memory_total_mb"] = mem.total // (1024 * 1024)
            info["memory_used_mb"] = mem.used // (1024 * 1024)
            info["memory_available_mb"] = mem.available // (1024 * 1024)
            info["memory_percent"] = mem.percent
        except Exception:
            info["memory_percent"] = None

        try:
            disk = psutil.disk_usage("/")
            info["disk_total_gb"] = round(disk.total / (1024 ** 3), 1)
            info["disk_used_gb"] = round(disk.used / (1024 ** 3), 1)
            info["disk_percent"] = disk.percent
        except Exception:
            info["disk_percent"] = None

        try:
            if hasattr(psutil, "getloadavg"):
                load = psutil.getloadavg()
                info["load_avg_1m"] = load[0]
                info["load_avg_5m"] = load[1]
                info["load_avg_15m"] = load[2]
        except Exception:
            pass

        try:
            proc = psutil.Process(os.getpid())
            info["process_memory_mb"] = round(proc.memory_info().rss / (1024 * 1024), 1)
            info["process_cpu_percent"] = proc.cpu_percent(interval=0.1)
            info["process_threads"] = proc.num_threads()
        except Exception:
            pass

        return info


# ===========================================================================
# Metric Recording Helpers
# ===========================================================================

def record_chat_request(
    model: str,
    provider: str,
    duration_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    success: bool,
) -> None:
    """Record metrics for a chat request."""
    status = "success" if success else "error"
    CHAT_REQUESTS_TOTAL.labels(model=model, provider=provider, status=status).inc()
    CHAT_DURATION_SECONDS.labels(model=model, provider=provider).observe(duration_seconds)
    TOKEN_USAGE_TOTAL.labels(model=model, token_type="prompt").inc(prompt_tokens)
    TOKEN_USAGE_TOTAL.labels(model=model, token_type="completion").inc(completion_tokens)
    COST_USD_TOTAL.labels(model=model, provider=provider).inc(cost_usd)


def record_cache_hit(hit_type: str = "semantic") -> None:
    """Record a semantic cache hit."""
    RAG_CACHE_HITS_TOTAL.labels(hit_type=hit_type).inc()


def record_rag_request(success: bool) -> None:
    """Record a RAG query."""
    RAG_REQUESTS_TOTAL.labels(status="success" if success else "error").inc()


def record_agent_run(status: str, iterations: int) -> None:
    """Record an agent run."""
    AGENT_RUNS_TOTAL.labels(status=status).inc()
    AGENT_ITERATIONS_HISTOGRAM.observe(iterations)


def record_memory_stored(memory_type: str) -> None:
    """Record a memory being stored."""
    MEMORY_STORE_TOTAL.labels(memory_type=memory_type).inc()


def set_active_users(count: int) -> None:
    """Update active users gauge."""
    ACTIVE_USERS_GAUGE.set(count)


def record_ws_connection(delta: int = 1) -> None:
    """Track WebSocket connections (+1 connect, -1 disconnect)."""
    WS_CONNECTIONS_GAUGE.inc(delta)


def record_ws_message(event_type: str) -> None:
    """Track WebSocket messages by event type."""
    WS_MESSAGES_TOTAL.labels(event_type=event_type).inc()


# ===========================================================================
# Singleton & Factory
# ===========================================================================
_monitoring_instance: Optional[Monitoring] = None


def init_monitoring(app: Flask) -> Monitoring:
    """
    Initialize and return the global Monitoring instance.
    Call once from app factory.
    """
    global _monitoring_instance
    if _monitoring_instance is None:
        _monitoring_instance = Monitoring(app)
    else:
        _monitoring_instance.init_app(app)
    return _monitoring_instance


def get_monitoring() -> Optional[Monitoring]:
    """Get the global Monitoring instance."""
    return _monitoring_instance


# Backward compatible singleton
monitoring = Monitoring()