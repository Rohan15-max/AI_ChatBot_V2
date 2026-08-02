"""
DevMentor AI — Analytics Service
===================================
Production-grade analytics with:

- Non-blocking usage tracking via background threads
- Real-time cost calculation with configurable pricing
- User-level and platform-level statistics
- Daily/weekly/monthly aggregations
- Cost projections and trend analysis
- Model performance metrics
- Agent run tracking
- Cache hit rate analytics
- Error rate monitoring
- Flask route registration
- Admin dashboard data

Usage:
    from analytics import get_analytics

    analytics = get_analytics()

    # Track LLM usage
    analytics.track_llm_usage(
        user_id="u123",
        model="gemini-flash",
        provider="gemini",
        prompt_tokens=100,
        completion_tokens=200,
        duration_ms=1500,
        cost_usd=0.0001,
    )

    # Get user stats
    stats = analytics.get_user_stats(user_id, days=30)
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, and_, distinct, case

from config import get_settings
from database import Message, Thread, UsageLog, User, get_db, utc_now

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Model pricing (USD per 1K tokens)
# ---------------------------------------------------------------------------
DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "gemini-flash":    {"input": 0.000075,  "output": 0.0003},
    "gemini-pro":      {"input": 0.00125,   "output": 0.005},
    "gpt-4o-mini":     {"input": 0.00015,   "output": 0.0006},
    "gpt-4o":          {"input": 0.0025,    "output": 0.01},
    "claude-haiku":    {"input": 0.00025,   "output": 0.00125},
    "claude-sonnet":   {"input": 0.003,     "output": 0.015},
    "ollama":          {"input": 0.0,       "output": 0.0},
    "default":         {"input": 0.0001,    "output": 0.0003},
}

# Allow price override via environment variable
_price_override = os.getenv("MODEL_PRICE_JSON")
if _price_override:
    try:
        DEFAULT_PRICES.update(json.loads(_price_override))
        logger.info("Model prices updated from MODEL_PRICE_JSON")
    except json.JSONDecodeError:
        logger.warning("MODEL_PRICE_JSON is invalid JSON — using defaults")


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate exact cost for a request.

    Args:
        model:        Model name
        input_tokens:  Prompt token count
        output_tokens: Completion token count

    Returns:
        Cost in USD
    """
    prices = DEFAULT_PRICES.get(model, DEFAULT_PRICES["default"])
    input_cost = prices["input"] * input_tokens / 1000
    output_cost = prices["output"] * output_tokens / 1000
    return input_cost + output_cost


# ===========================================================================
# Analytics Service
# ===========================================================================

class Analytics:
    """
    Production-grade analytics service.

    All write operations are non-blocking (background threads).
    All read operations are synchronous (for simplicity).
    """

    def __init__(self):
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Tracking Methods (non-blocking)
    # -----------------------------------------------------------------------

    def track_llm_usage(
        self,
        user_id: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
        cost_usd: float = 0.0,
        endpoint: str = "chat",
        thread_id: Optional[str] = None,
        had_error: bool = False,
        is_cached: bool = False,
        cache_hit_type: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Track LLM API usage asynchronously.

        Args:
            user_id:           User identifier
            model:             Model name used
            provider:          LLM provider name
            prompt_tokens:     Input token count
            completion_tokens: Output token count
            duration_ms:       Response time in milliseconds
            cost_usd:          Actual cost (0 = auto-calculate)
            endpoint:          API endpoint called
            thread_id:         Conversation thread ID
            had_error:         Whether request failed
            is_cached:         Whether response was cached
            cache_hit_type:    exact | semantic | none
            request_id:        Unique request trace ID

        Returns:
            Dict with tokens and cost (immediate estimate)
        """
        total_tokens = prompt_tokens + completion_tokens

        # Auto-calculate cost if not provided
        if cost_usd == 0.0 and not is_cached:
            cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)

        # Fire-and-forget background write
        threading.Thread(
            target=self._write_usage_log,
            args=(
                user_id, endpoint, model, provider,
                prompt_tokens, completion_tokens, total_tokens,
                cost_usd, duration_ms, thread_id,
                had_error, is_cached, cache_hit_type, request_id,
            ),
            daemon=True,
        ).start()

        return {
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 6),
            "model": model,
        }

    def track_agent_run(
        self,
        user_id: str,
        run_id: str,
        status: str,
        iterations: int,
        total_tokens: int,
        total_cost_usd: float,
        duration_ms: int,
    ) -> None:
        """Track an agent execution run."""
        threading.Thread(
            target=self._write_agent_log,
            args=(user_id, run_id, status, iterations, total_tokens, total_cost_usd, duration_ms),
            daemon=True,
        ).start()

    def track_error(
        self,
        user_id: str,
        endpoint: str,
        error_type: str,
        duration_ms: int = 0,
    ) -> None:
        """Track an API error."""
        threading.Thread(
            target=self._write_usage_log,
            args=(
                user_id, endpoint, "unknown", "unknown",
                0, 0, 0, 0.0, duration_ms, None,
                True, False, None, None,
            ),
            daemon=True,
        ).start()

    # -----------------------------------------------------------------------
    # Private Write Methods
    # -----------------------------------------------------------------------

    def _write_usage_log(
        self,
        user_id: str,
        endpoint: str,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
        duration_ms: int,
        thread_id: Optional[str],
        had_error: bool,
        is_cached: bool,
        cache_hit_type: Optional[str],
        request_id: Optional[str],
    ) -> None:
        """Write usage log to database (runs in background thread)."""
        try:
            with get_db() as db:
                log = UsageLog(
                    user_id=user_id,
                    endpoint=endpoint,
                    model=model,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                    thread_id=thread_id,
                    had_error=had_error,
                    is_cached=is_cached,
                    cache_hit_type=cache_hit_type,
                    request_id=request_id,
                    status_code=200 if not had_error else 500,
                )
                db.add(log)

                # Update user token counters
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    user.total_tokens_used = (user.total_tokens_used or 0) + total_tokens
                    user.monthly_tokens_used = (user.monthly_tokens_used or 0) + total_tokens

        except Exception as exc:
            logger.error("Failed to write usage log: %s", exc)

    def _write_agent_log(
        self,
        user_id: str,
        run_id: str,
        status: str,
        iterations: int,
        total_tokens: int,
        total_cost_usd: float,
        duration_ms: int,
    ) -> None:
        """Write agent run log (runs in background thread)."""
        try:
            with get_db() as db:
                log = UsageLog(
                    user_id=user_id,
                    endpoint="/agent",
                    model="agent",
                    provider="multi",
                    prompt_tokens=total_tokens // 2,
                    completion_tokens=total_tokens // 2,
                    total_tokens=total_tokens,
                    cost_usd=total_cost_usd,
                    duration_ms=duration_ms,
                    had_error=(status == "failed"),
                    request_id=run_id,
                )
                db.add(log)
        except Exception as exc:
            logger.error("Failed to write agent log: %s", exc)

    # -----------------------------------------------------------------------
    # User Statistics
    # -----------------------------------------------------------------------

    def get_user_stats(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics for a user.

        Args:
            user_id: User identifier
            days:    Lookback window in days

        Returns:
            Dict with tokens, costs, requests, model breakdown, daily usage
        """
        if not user_id:
            return {}

        since = utc_now() - timedelta(days=days)

        try:
            with get_db() as db:
                # ---- Totals ----
                totals = db.query(
                    func.sum(UsageLog.total_tokens).label("total_tokens"),
                    func.sum(UsageLog.cost_usd).label("total_cost"),
                    func.count(UsageLog.id).label("total_requests"),
                    func.avg(UsageLog.duration_ms).label("avg_latency"),
                    func.sum(case((UsageLog.had_error == True, 1), else_=0)).label("error_count"),
                ).filter(
                    UsageLog.user_id == user_id,
                    UsageLog.created_at >= since,
                ).first()

                # ---- By model ----
                by_model = db.query(
                    UsageLog.model,
                    func.sum(UsageLog.total_tokens).label("tokens"),
                    func.sum(UsageLog.cost_usd).label("cost"),
                    func.count(UsageLog.id).label("requests"),
                ).filter(
                    UsageLog.user_id == user_id,
                    UsageLog.created_at >= since,
                ).group_by(UsageLog.model).all()

                # ---- Daily usage ----
                daily = db.query(
                    func.date(UsageLog.created_at).label("date"),
                    func.sum(UsageLog.total_tokens).label("tokens"),
                    func.sum(UsageLog.cost_usd).label("cost"),
                    func.count(UsageLog.id).label("requests"),
                ).filter(
                    UsageLog.user_id == user_id,
                    UsageLog.created_at >= since,
                ).group_by(func.date(UsageLog.created_at)).order_by("date").all()

                # ---- Conversation stats ----
                thread_count = db.query(func.count(Thread.id)).filter(
                    Thread.user_id == user_id,
                    Thread.is_deleted == False,
                ).scalar() or 0

                message_count = db.query(func.count(Message.id)).join(Thread).filter(
                    Thread.user_id == user_id,
                ).scalar() or 0

                # ---- Cache stats ----
                cache_hits = db.query(func.count(UsageLog.id)).filter(
                    UsageLog.user_id == user_id,
                    UsageLog.created_at >= since,
                    UsageLog.is_cached == True,
                ).scalar() or 0

                total_requests = totals.total_requests or 0
                cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

                return {
                    "user_id": user_id,
                    "period_days": days,
                    "total_tokens": totals.total_tokens or 0,
                    "total_cost_usd": round(float(totals.total_cost or 0), 6),
                    "total_requests": total_requests,
                    "avg_latency_ms": round(float(totals.avg_latency or 0), 1),
                    "error_count": int(totals.error_count or 0),
                    "error_rate_percent": round(
                        (int(totals.error_count or 0) / total_requests * 100)
                        if total_requests > 0 else 0, 1
                    ),
                    "cache_hits": cache_hits,
                    "cache_hit_rate_percent": round(cache_hit_rate, 1),
                    "by_model": [
                        {
                            "model": m.model,
                            "tokens": m.tokens or 0,
                            "cost_usd": round(float(m.cost or 0), 6),
                            "requests": m.requests or 0,
                        }
                        for m in by_model
                    ],
                    "daily_usage": [
                        {
                            "date": str(d.date),
                            "tokens": d.tokens or 0,
                            "cost_usd": round(float(d.cost or 0), 6),
                            "requests": d.requests or 0,
                        }
                        for d in daily
                    ],
                    "thread_count": thread_count,
                    "message_count": message_count,
                }

        except Exception as exc:
            logger.error("get_user_stats failed for user %s: %s", user_id, exc)
            return {"user_id": user_id, "error": str(exc)}

    def get_cost_projection(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Project future costs based on historical usage.

        Args:
            user_id: User identifier
            days:    Historical window for projection

        Returns:
            Dict with daily average, monthly projection, and estimated cost
        """
        stats = self.get_user_stats(user_id, days=min(days, 90))

        total_tokens = stats.get("total_tokens", 0)
        total_cost = stats.get("total_cost_usd", 0.0)
        actual_days = min(days, 90)

        daily_avg_tokens = total_tokens / actual_days if actual_days > 0 else 0
        daily_avg_cost = total_cost / actual_days if actual_days > 0 else 0

        return {
            "user_id": user_id,
            "based_on_days": actual_days,
            "daily_avg_tokens": round(daily_avg_tokens),
            "daily_avg_cost_usd": round(daily_avg_cost, 6),
            "monthly_projection_tokens": round(daily_avg_tokens * 30),
            "monthly_projection_cost_usd": round(daily_avg_cost * 30, 4),
            "yearly_projection_cost_usd": round(daily_avg_cost * 365, 2),
        }

    # -----------------------------------------------------------------------
    # Platform Statistics (Admin)
    # -----------------------------------------------------------------------

    def get_platform_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Global platform statistics for admin dashboard.

        Args:
            days: Lookback window

        Returns:
            Dict with user counts, usage, costs, model distribution
        """
        since = utc_now() - timedelta(days=days)

        try:
            with get_db() as db:
                # User metrics
                total_users = db.query(func.count(User.id)).filter(
                    User.is_deleted == False
                ).scalar() or 0

                active_users = db.query(
                    func.count(distinct(UsageLog.user_id))
                ).filter(UsageLog.created_at >= since).scalar() or 0

                new_users = db.query(func.count(User.id)).filter(
                    User.created_at >= since,
                    User.is_deleted == False,
                ).scalar() or 0

                # Usage metrics
                usage_totals = db.query(
                    func.sum(UsageLog.total_tokens).label("tokens"),
                    func.sum(UsageLog.cost_usd).label("cost"),
                    func.count(UsageLog.id).label("requests"),
                    func.avg(UsageLog.duration_ms).label("avg_latency"),
                ).filter(UsageLog.created_at >= since).first()

                # Error rate
                error_count = db.query(func.count(UsageLog.id)).filter(
                    UsageLog.created_at >= since,
                    UsageLog.had_error == True,
                ).scalar() or 0

                total_requests = usage_totals.requests or 0
                error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0

                # Tier distribution
                tiers = db.query(
                    User.tier, func.count(User.id)
                ).filter(
                    User.is_deleted == False
                ).group_by(User.tier).all()

                # Model distribution
                models = db.query(
                    UsageLog.model,
                    func.count(UsageLog.id).label("requests"),
                    func.sum(UsageLog.total_tokens).label("tokens"),
                ).filter(
                    UsageLog.created_at >= since
                ).group_by(UsageLog.model).order_by(
                    func.count(UsageLog.id).desc()
                ).limit(10).all()

                # Cache hit rate
                cache_hits = db.query(func.count(UsageLog.id)).filter(
                    UsageLog.created_at >= since,
                    UsageLog.is_cached == True,
                ).scalar() or 0

                cache_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

                # Daily trend
                daily = db.query(
                    func.date(UsageLog.created_at).label("date"),
                    func.count(UsageLog.id).label("requests"),
                    func.sum(UsageLog.cost_usd).label("cost"),
                    func.count(distinct(UsageLog.user_id)).label("active_users"),
                ).filter(
                    UsageLog.created_at >= since
                ).group_by(func.date(UsageLog.created_at)).order_by("date").all()

                return {
                    "period_days": days,
                    "users": {
                        "total": total_users,
                        "active_in_period": active_users,
                        "new_in_period": new_users,
                        "tier_distribution": {t[0]: t[1] for t in tiers},
                    },
                    "usage": {
                        "total_tokens": usage_totals.tokens or 0,
                        "total_cost_usd": round(float(usage_totals.cost or 0), 4),
                        "total_requests": total_requests,
                        "avg_latency_ms": round(float(usage_totals.avg_latency or 0), 1),
                        "error_count": error_count,
                        "error_rate_percent": round(error_rate, 2),
                        "cache_hit_rate_percent": round(cache_rate, 1),
                    },
                    "models": [
                        {
                            "model": m.model,
                            "requests": m.requests or 0,
                            "tokens": m.tokens or 0,
                        }
                        for m in models
                    ],
                    "daily_trend": [
                        {
                            "date": str(d.date),
                            "requests": d.requests or 0,
                            "cost_usd": round(float(d.cost or 0), 4),
                            "active_users": d.active_users or 0,
                        }
                        for d in daily
                    ],
                    "generated_at": utc_now().isoformat(),
                }

        except Exception as exc:
            logger.error("get_platform_stats failed: %s", exc)
            return {"error": str(exc)}

    def get_user_leaderboard(
        self,
        metric: str = "tokens",
        limit: int = 10,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Top users by tokens, cost, or requests.

        Args:
            metric: tokens | cost | requests
            limit:  Number of users to return
            days:   Lookback window

        Returns:
            List of user dicts sorted by metric descending
        """
        since = utc_now() - timedelta(days=days)

        if metric not in ("tokens", "cost", "requests"):
            return []

        try:
            with get_db() as db:
                if metric == "tokens":
                    order_col = func.sum(UsageLog.total_tokens)
                    value_col = func.sum(UsageLog.total_tokens).label("value")
                elif metric == "cost":
                    order_col = func.sum(UsageLog.cost_usd)
                    value_col = func.sum(UsageLog.cost_usd).label("value")
                else:
                    order_col = func.count(UsageLog.id)
                    value_col = func.count(UsageLog.id).label("value")

                results = db.query(
                    User.username,
                    User.tier,
                    value_col,
                ).join(
                    UsageLog, User.id == UsageLog.user_id
                ).filter(
                    UsageLog.created_at >= since,
                ).group_by(User.id, User.username, User.tier).order_by(
                    order_col.desc()
                ).limit(limit).all()

                return [
                    {
                        "rank": i + 1,
                        "username": r.username,
                        "tier": r.tier,
                        "value": round(float(r.value or 0), 6),
                        "metric": metric,
                    }
                    for i, r in enumerate(results)
                ]

        except Exception as exc:
            logger.error("get_user_leaderboard failed: %s", exc)
            return []

    def get_model_performance(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get performance metrics per model for monitoring dashboard.

        Returns:
            List of model metrics with latency, error rate, usage
        """
        since = utc_now() - timedelta(days=days)

        try:
            with get_db() as db:
                results = db.query(
                    UsageLog.model,
                    UsageLog.provider,
                    func.count(UsageLog.id).label("requests"),
                    func.avg(UsageLog.duration_ms).label("avg_latency"),
                    func.min(UsageLog.duration_ms).label("min_latency"),
                    func.max(UsageLog.duration_ms).label("max_latency"),
                    func.sum(UsageLog.total_tokens).label("total_tokens"),
                    func.sum(UsageLog.cost_usd).label("total_cost"),
                ).filter(
                    UsageLog.created_at >= since,
                    UsageLog.model != "unknown",
                ).group_by(
                    UsageLog.model, UsageLog.provider
                ).order_by(func.count(UsageLog.id).desc()).all()

                return [
                    {
                        "model": r.model,
                        "provider": r.provider,
                        "requests": r.requests or 0,
                        "avg_latency_ms": round(float(r.avg_latency or 0), 1),
                        "min_latency_ms": r.min_latency or 0,
                        "max_latency_ms": r.max_latency or 0,
                        "total_tokens": r.total_tokens or 0,
                        "total_cost_usd": round(float(r.total_cost or 0), 4),
                    }
                    for r in results
                ]

        except Exception as exc:
            logger.error("get_model_performance failed: %s", exc)
            return []


# ===========================================================================
# Singleton
# ===========================================================================
_analytics_instance: Optional[Analytics] = None


def get_analytics() -> Analytics:
    """Get or create the global Analytics singleton."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = Analytics()
    return _analytics_instance


# Backward compatible singleton
analytics = get_analytics()


# ===========================================================================
# Flask Route Registration
# ===========================================================================

def register_analytics_routes(app) -> None:
    """
    Register analytics API routes with Flask app.
    Call from app factory.
    """
    from flask import jsonify, request
    from auth_middleware import require_auth, require_admin

    @app.route("/api/stats/me", methods=["GET"])
    @require_auth
    def get_my_stats():
        days = min(int(request.args.get("days", 30)), 90)
        stats = get_analytics().get_user_stats(request.user_id, days=days)
        return jsonify({"success": True, "data": stats})

    @app.route("/api/stats/me/projection", methods=["GET"])
    @require_auth
    def get_cost_projection():
        days = min(int(request.args.get("days", 30)), 90)
        projection = get_analytics().get_cost_projection(request.user_id, days=days)
        return jsonify({"success": True, "data": projection})

    @app.route("/api/stats/platform", methods=["GET"])
    @require_admin
    def get_platform_stats():
        days = min(int(request.args.get("days", 30)), 90)
        stats = get_analytics().get_platform_stats(days=days)
        return jsonify({"success": True, "data": stats})

    @app.route("/api/stats/leaderboard", methods=["GET"])
    @require_admin
    def get_leaderboard():
        metric = request.args.get("metric", "tokens")
        limit = min(int(request.args.get("limit", 10)), 50)
        data = get_analytics().get_user_leaderboard(metric=metric, limit=limit)
        return jsonify({"success": True, "data": data})

    @app.route("/api/stats/models", methods=["GET"])
    @require_admin
    def get_model_performance():
        days = min(int(request.args.get("days", 7)), 30)
        data = get_analytics().get_model_performance(days=days)
        return jsonify({"success": True, "data": data})

    logger.info("Analytics routes registered")