"""
DevMentor AI — Admin Panel
============================
Production-grade admin panel with:

- User management (list, update, deactivate, reactivate, delete)
- Platform analytics and usage statistics
- Model performance metrics
- Audit log viewer
- Rate limit management
- Feature flag controls
- System health overview
- CSV export
- Proper admin-only access via is_admin flag (not tier string)

Usage:
    from admin_panel import admin_bp

    # In app factory
    app.register_blueprint(admin_bp)
"""

import csv
import logging
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Dict, Optional

from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import func, distinct

from auth_middleware import require_admin
from analytics import get_analytics
from config import get_settings
from database import AuditLog, Message, Thread, UsageLog, User, get_db, utc_now

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# ---------------------------------------------------------------------------
# Allowed values
# ---------------------------------------------------------------------------
ALLOWED_TIERS = {"free", "pro", "enterprise"}
MAX_PAGE_SIZE = 100


# ===========================================================================
# Helpers
# ===========================================================================

def _write_audit_log(
    action: str,
    target_id: Optional[str] = None,
    details: Optional[Dict] = None,
) -> None:
    """Write admin action to audit log."""
    try:
        admin_id = getattr(g, "user_id", None) or getattr(request, "user_id", None)
        with get_db() as db:
            entry = AuditLog(
                user_id=admin_id,
                action=f"admin.{action}",
                resource_type="user" if target_id else None,
                resource_id=target_id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent", "")[:300],
                success=True,
            )
            db.add(entry)
        logger.info(
            "Admin action: %s by %s on %s details=%s",
            action, admin_id, target_id, details,
        )
    except Exception as exc:
        logger.error("Failed to write admin audit log: %s", exc)


def _paginate(query, page: int, per_page: int):
    """Apply pagination to a SQLAlchemy query."""
    per_page = min(per_page, MAX_PAGE_SIZE)
    total = query.count()
    items = query.offset((page - 1) * per_page).limit(per_page).all()
    return items, {
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": max(1, (total + per_page - 1) // per_page),
    }


# ===========================================================================
# Dashboard
# ===========================================================================

@admin_bp.route("/")
@admin_bp.route("/dashboard")
@require_admin
def dashboard():
    """Admin dashboard — platform overview stats."""
    stats = get_analytics().get_platform_stats(days=30)
    return jsonify({"success": True, "data": stats})


@admin_bp.route("/health")
@require_admin
def system_health():
    """System health overview for admin panel."""
    try:
        from monitoring import get_monitoring
        monitoring = get_monitoring()
        if monitoring:
            db_status = monitoring._check_database()
            redis_status = monitoring._check_redis()
            qdrant_status = monitoring._check_qdrant()
        else:
            db_status = redis_status = qdrant_status = {"healthy": False, "error": "monitoring not initialized"}
    except Exception as exc:
        db_status = redis_status = qdrant_status = {"healthy": False, "error": str(exc)}

    return jsonify({
        "success": True,
        "data": {
            "database": db_status,
            "redis": redis_status,
            "qdrant": qdrant_status,
            "timestamp": utc_now().isoformat(),
        },
    })


# ===========================================================================
# User Management
# ===========================================================================

@admin_bp.route("/api/users", methods=["GET"])
@require_admin
def list_users():
    """
    List all users with pagination and search.

    Query params:
        page:     Page number (default: 1)
        per_page: Results per page (max: 100)
        search:   Search username or email
        tier:     Filter by tier
        active:   Filter by active status (true/false)
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    search = request.args.get("search", "").strip()
    tier_filter = request.args.get("tier", "").strip()
    active_filter = request.args.get("active", "").strip()

    with get_db() as db:
        query = db.query(User).filter(User.is_deleted == False)

        if search:
            query = query.filter(
                User.username.ilike(f"%{search}%") |
                User.email.ilike(f"%{search}%")
            )

        if tier_filter in ALLOWED_TIERS:
            query = query.filter(User.tier == tier_filter)

        if active_filter.lower() == "true":
            query = query.filter(User.is_active == True)
        elif active_filter.lower() == "false":
            query = query.filter(User.is_active == False)

        query = query.order_by(User.created_at.desc())
        users, pagination = _paginate(query, page, per_page)

        return jsonify({
            "success": True,
            "data": [
                {
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "tier": u.tier,
                    "is_active": u.is_active,
                    "is_admin": u.is_admin,
                    "is_verified": u.is_verified,
                    "total_tokens_used": u.total_tokens_used,
                    "monthly_tokens_used": u.monthly_tokens_used,
                    "login_count": u.login_count,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "last_active": u.last_active.isoformat() if u.last_active else None,
                    "last_login": u.last_login.isoformat() if u.last_login else None,
                }
                for u in users
            ],
            "pagination": pagination,
        })


@admin_bp.route("/api/users/<user_id>", methods=["GET"])
@require_admin
def get_user(user_id: str):
    """Get detailed info for a specific user."""
    with get_db() as db:
        user = db.query(User).filter(
            User.id == user_id,
            User.is_deleted == False,
        ).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get usage stats
        stats = get_analytics().get_user_stats(user_id, days=30)

        return jsonify({
            "success": True,
            "data": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "tier": user.tier,
                "is_active": user.is_active,
                "is_admin": user.is_admin,
                "is_verified": user.is_verified,
                "total_tokens_used": user.total_tokens_used,
                "monthly_tokens_used": user.monthly_tokens_used,
                "login_count": user.login_count,
                "failed_login_attempts": user.failed_login_attempts,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_active": user.last_active.isoformat() if user.last_active else None,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "usage_stats": stats,
            },
        })


@admin_bp.route("/api/users/<user_id>", methods=["PUT"])
@require_admin
def update_user(user_id: str):
    """
    Update a user's tier, active status, or admin flag.

    Body:
        tier:     free | pro | enterprise
        is_active: bool
        is_admin:  bool
    """
    data = request.get_json(silent=True) or {}
    tier = data.get("tier")
    is_active = data.get("is_active")
    is_admin = data.get("is_admin")

    # Validation
    if tier is not None and tier not in ALLOWED_TIERS:
        return jsonify({"error": f"Invalid tier. Allowed: {', '.join(ALLOWED_TIERS)}"}), 400

    if is_active is not None and not isinstance(is_active, bool):
        return jsonify({"error": "is_active must be boolean"}), 400

    if is_admin is not None and not isinstance(is_admin, bool):
        return jsonify({"error": "is_admin must be boolean"}), 400

    # Prevent self-demotion
    admin_id = getattr(g, "user_id", None)
    if user_id == admin_id and is_admin is False:
        return jsonify({"error": "Cannot remove your own admin privileges"}), 400

    with get_db() as db:
        user = db.query(User).filter(
            User.id == user_id,
            User.is_deleted == False,
        ).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        old_values = {
            "tier": user.tier,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
        }

        if tier is not None:
            user.tier = tier
        if is_active is not None:
            user.is_active = is_active
        if is_admin is not None:
            user.is_admin = is_admin

    _write_audit_log("update_user", user_id, {"old": old_values, "new": data})
    return jsonify({"success": True, "message": "User updated successfully"})


@admin_bp.route("/api/users/<user_id>", methods=["DELETE"])
@require_admin
def deactivate_user(user_id: str):
    """Soft-deactivate a user (set is_active=False)."""
    admin_id = getattr(g, "user_id", None)
    if user_id == admin_id:
        return jsonify({"error": "Cannot deactivate your own account"}), 400

    with get_db() as db:
        user = db.query(User).filter(
            User.id == user_id,
            User.is_deleted == False,
        ).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        user.is_active = False

    _write_audit_log("deactivate_user", user_id, {"username": user.username})
    return jsonify({"success": True, "message": "User deactivated"})


@admin_bp.route("/api/users/<user_id>/reactivate", methods=["POST"])
@require_admin
def reactivate_user(user_id: str):
    """Reactivate a deactivated user."""
    with get_db() as db:
        user = db.query(User).filter(
            User.id == user_id,
            User.is_deleted == False,
        ).first()

        if not user:
            return jsonify({"error": "User not found"}), 404

        user.is_active = True

    _write_audit_log("reactivate_user", user_id)
    return jsonify({"success": True, "message": "User reactivated"})


@admin_bp.route("/api/users/<user_id>/revoke-tokens", methods=["POST"])
@require_admin
def revoke_user_tokens(user_id: str):
    """Revoke all active JWT tokens for a user."""
    try:
        from security.jwt_blacklist import revoke_all_user_tokens
        success = revoke_all_user_tokens(user_id, reason="admin_revocation")
        if success:
            _write_audit_log("revoke_tokens", user_id)
            return jsonify({"success": True, "message": "All tokens revoked"})
        return jsonify({"error": "Failed to revoke tokens"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route("/api/users/<user_id>/reset-rate-limit", methods=["POST"])
@require_admin
def reset_rate_limit(user_id: str):
    """Reset rate limit counters for a user."""
    try:
        from rate_limiter import reset_user_rate_limit
        reset_user_rate_limit(user_id)
        _write_audit_log("reset_rate_limit", user_id)
        return jsonify({"success": True, "message": "Rate limits reset"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@admin_bp.route("/api/users/<user_id>/delete-memories", methods=["DELETE"])
@require_admin
def delete_user_memories(user_id: str):
    """Delete all long-term memories for a user (GDPR)."""
    import asyncio
    try:
        from long_term_memory import get_long_term_memory
        loop = asyncio.new_event_loop()
        count = loop.run_until_complete(
            get_long_term_memory().delete_user_data(user_id)
        )
        loop.close()
        _write_audit_log("delete_memories", user_id, {"count": count})
        return jsonify({"success": True, "deleted": count})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ===========================================================================
# Analytics
# ===========================================================================

@admin_bp.route("/api/usage", methods=["GET"])
@require_admin
def get_usage():
    """Platform usage statistics with daily breakdown."""
    days = min(request.args.get("days", 30, type=int), 365)
    stats = get_analytics().get_platform_stats(days=days)
    return jsonify({"success": True, "data": stats})


@admin_bp.route("/api/models/performance", methods=["GET"])
@require_admin
def model_performance():
    """Per-model performance metrics."""
    days = min(request.args.get("days", 7, type=int), 90)
    data = get_analytics().get_model_performance(days=days)
    return jsonify({"success": True, "data": data})


@admin_bp.route("/api/leaderboard", methods=["GET"])
@require_admin
def leaderboard():
    """Top users by tokens, cost, or requests."""
    metric = request.args.get("metric", "tokens")
    limit = min(request.args.get("limit", 10, type=int), 50)
    days = min(request.args.get("days", 30, type=int), 90)
    data = get_analytics().get_user_leaderboard(metric=metric, limit=limit, days=days)
    return jsonify({"success": True, "data": data})


# ===========================================================================
# Audit Logs
# ===========================================================================

@admin_bp.route("/api/audit-logs", methods=["GET"])
@require_admin
def get_audit_logs():
    """
    View audit logs with filtering.

    Query params:
        page, per_page, user_id, action, days
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    user_id_filter = request.args.get("user_id", "").strip()
    action_filter = request.args.get("action", "").strip()
    days = min(request.args.get("days", 30, type=int), 365)

    since = utc_now() - timedelta(days=days)

    with get_db() as db:
        query = db.query(AuditLog).filter(AuditLog.created_at >= since)

        if user_id_filter:
            query = query.filter(AuditLog.user_id == user_id_filter)

        if action_filter:
            query = query.filter(AuditLog.action.ilike(f"%{action_filter}%"))

        query = query.order_by(AuditLog.created_at.desc())
        logs, pagination = _paginate(query, page, per_page)

        return jsonify({
            "success": True,
            "data": [
                {
                    "id": log.id,
                    "user_id": log.user_id,
                    "action": log.action,
                    "resource_type": log.resource_type,
                    "resource_id": log.resource_id,
                    "ip_address": log.ip_address,
                    "success": log.success,
                    "failure_reason": log.failure_reason,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in logs
            ],
            "pagination": pagination,
        })


# ===========================================================================
# Feature Flags
# ===========================================================================

@admin_bp.route("/api/features", methods=["GET"])
@require_admin
def get_features():
    """Get current feature flag states."""
    return jsonify({
        "success": True,
        "data": {
            "streaming": settings.feature_streaming,
            "rag": settings.feature_rag,
            "agent": settings.feature_agent,
            "long_term_memory": settings.feature_long_term_memory,
            "semantic_cache": settings.feature_semantic_cache,
            "google_oauth": settings.feature_google_oauth,
            "admin_panel": settings.feature_admin_panel,
            "webhooks": settings.feature_webhooks,
            "analytics": settings.feature_analytics,
            "multimodal": settings.feature_multimodal,
        },
    })


# ===========================================================================
# Export
# ===========================================================================

@admin_bp.route("/api/export/users", methods=["GET"])
@require_admin
def export_users_csv():
    """Export all users as CSV."""
    with get_db() as db:
        users = db.query(User).filter(
            User.is_deleted == False
        ).order_by(User.created_at).all()

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "username", "email", "tier", "is_active", "is_admin",
            "total_tokens_used", "login_count",
            "created_at", "last_active", "last_login",
        ])
        for u in users:
            writer.writerow([
                u.id, u.username, u.email or "",
                u.tier, u.is_active, u.is_admin,
                u.total_tokens_used or 0, u.login_count or 0,
                u.created_at.isoformat() if u.created_at else "",
                u.last_active.isoformat() if u.last_active else "",
                u.last_login.isoformat() if u.last_login else "",
            ])

    _write_audit_log("export_users_csv", details={"count": len(users)})

    return current_app.response_class(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=users_{datetime.now().strftime('%Y%m%d')}.csv"
        },
    )


@admin_bp.route("/api/export/usage", methods=["GET"])
@require_admin
def export_usage_csv():
    """Export usage logs as CSV."""
    days = min(request.args.get("days", 30, type=int), 90)
    since = utc_now() - timedelta(days=days)

    with get_db() as db:
        logs = db.query(UsageLog).filter(
            UsageLog.created_at >= since
        ).order_by(UsageLog.created_at.desc()).limit(10000).all()

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "user_id", "endpoint", "model", "provider",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cost_usd", "duration_ms", "had_error", "is_cached", "created_at",
        ])
        for log in logs:
            writer.writerow([
                log.id, log.user_id, log.endpoint, log.model, log.provider,
                log.prompt_tokens, log.completion_tokens, log.total_tokens,
                log.cost_usd, log.duration_ms, log.had_error, log.is_cached,
                log.created_at.isoformat() if log.created_at else "",
            ])

    return current_app.response_class(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=usage_{datetime.now().strftime('%Y%m%d')}.csv"
        },
    )