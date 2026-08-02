"""
DevMentor AI — Audit Logger
==============================
Production-grade security audit logging with:

- Dual-write: structured file log + database (AuditLog model)
- PII redaction before logging
- Async-safe (no blocking I/O in request path via background write option)
- Log rotation support (file-based)
- Immutable audit trail (append-only, never updated/deleted)
- Queryable audit events for admin panel
- Severity classification per event type
- Graceful degradation — never breaks the calling request

Usage:
    from security.audit_logger import log_audit

    log_audit(
        event_type="user.login",
        user_id="u123",
        resource="auth",
        success=True,
        details={"method": "password"},
    )
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_LOG_DIR = Path(os.getenv("AUDIT_LOG_DIR", "logs"))
AUDIT_LOG_PATH = _DEFAULT_LOG_DIR / "audit.log"
MAX_LOG_SIZE_BYTES = 50 * 1024 * 1024  # 50MB before rotation
_write_lock = threading.Lock()

# Event types considered security-critical (always logged at WARNING+ level)
CRITICAL_EVENTS = {
    "auth.failed", "auth.login_locked", "admin.access_denied",
    "user.deleted", "user.tier_downgraded", "api_key.revoked",
    "security.injection_detected", "security.rate_limit_abuse",
}


# ===========================================================================
# Setup
# ===========================================================================

def _ensure_log_dir() -> None:
    """Ensure the audit log directory exists."""
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Failed to create audit log directory: %s", exc)


def _rotate_if_needed() -> None:
    """Rotate the audit log file if it exceeds the max size."""
    try:
        if AUDIT_LOG_PATH.exists() and AUDIT_LOG_PATH.stat().st_size > MAX_LOG_SIZE_BYTES:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            rotated_path = AUDIT_LOG_PATH.with_name(f"audit_{timestamp}.log")
            AUDIT_LOG_PATH.rename(rotated_path)
            logger.info("Audit log rotated to %s", rotated_path)
    except Exception as exc:
        logger.error("Audit log rotation failed: %s", exc)


_ensure_log_dir()


# ===========================================================================
# PII Redaction Helper
# ===========================================================================

def _redact_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Redact PII from audit log details before persisting."""
    try:
        from security.pii_redactor import redact_dict
        return redact_dict(details)
    except Exception:
        return details


# ===========================================================================
# Core Logging Function
# ===========================================================================

def log_audit(
    event_type: str,
    user_id: Optional[str],
    resource: str,
    success: bool,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
    write_to_db: bool = True,
) -> Dict[str, Any]:
    """
    Log a security-relevant audit event.

    Writes to both the structured file log (always) and the database
    AuditLog table (if write_to_db=True and DB is reachable). File write
    is the source of truth that never fails silently into nothing —
    even if DB write fails, the file entry persists.

    Args:
        event_type:   Dotted event name e.g. "user.login", "admin.delete_user"
        user_id:      Acting user ID, or None for unauthenticated events
        resource:     Resource affected e.g. "auth", "thread:abc123"
        success:      Whether the action succeeded
        details:      Additional structured context (PII auto-redacted)
        ip_address:   Client IP address
        request_id:   Request trace ID for correlation
        write_to_db:  Whether to also persist to the database

    Returns:
        The logged entry dict (useful for tests/debugging)
    """
    details = _redact_details(details or {})

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "resource": resource,
        "success": success,
        "ip_address": ip_address,
        "request_id": request_id,
        "details": details,
        "severity": "critical" if event_type in CRITICAL_EVENTS else "info",
    }

    _write_to_file(entry)

    if write_to_db:
        _write_to_db_async(entry)

    if entry["severity"] == "critical":
        logger.warning("AUDIT[%s]: %s", event_type, json.dumps(entry, default=str))
    else:
        logger.info("AUDIT[%s]: user=%s resource=%s success=%s", event_type, user_id, resource, success)

    return entry


def _write_to_file(entry: Dict[str, Any]) -> None:
    """Append entry to the audit log file. Never raises."""
    try:
        with _write_lock:
            _rotate_if_needed()
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        # Audit logging must never crash the calling request
        logger.error("Failed to write audit log entry to file: %s", exc)


def _write_to_db_async(entry: Dict[str, Any]) -> None:
    """Write entry to database in a background thread — never blocks caller."""
    def _write():
        try:
            from database import AuditLog, get_db
            with get_db() as db:
                db.add(AuditLog(
                    user_id=entry.get("user_id"),
                    action=entry["event_type"],
                    resource_type=entry.get("resource", "").split(":")[0] if entry.get("resource") else None,
                    resource_id=entry.get("resource", "").split(":")[-1] if ":" in (entry.get("resource") or "") else None,
                    ip_address=entry.get("ip_address"),
                    request_id=entry.get("request_id"),
                    success=entry.get("success", True),
                    failure_reason=None if entry.get("success") else json.dumps(entry.get("details", {}))[:300],
                ))
        except Exception as exc:
            logger.error("Failed to write audit log entry to database: %s", exc)

    threading.Thread(target=_write, daemon=True).start()


# ===========================================================================
# Convenience Wrappers for Common Events
# ===========================================================================

def log_login(user_id: str, success: bool, ip_address: Optional[str] = None, method: str = "password") -> None:
    """Log a login attempt."""
    log_audit(
        event_type="auth.login" if success else "auth.failed",
        user_id=user_id,
        resource="auth",
        success=success,
        details={"method": method},
        ip_address=ip_address,
    )


def log_logout(user_id: str, ip_address: Optional[str] = None) -> None:
    """Log a logout event."""
    log_audit(
        event_type="auth.logout",
        user_id=user_id,
        resource="auth",
        success=True,
        ip_address=ip_address,
    )


def log_admin_action(
    admin_id: str,
    action: str,
    target_id: Optional[str] = None,
    details: Optional[Dict] = None,
    success: bool = True,
) -> None:
    """Log an admin panel action."""
    log_audit(
        event_type=f"admin.{action}",
        user_id=admin_id,
        resource=f"user:{target_id}" if target_id else "admin",
        success=success,
        details=details,
    )


def log_security_event(
    event_type: str,
    user_id: Optional[str],
    ip_address: Optional[str],
    details: Optional[Dict] = None,
) -> None:
    """Log a security-relevant event (injection attempt, rate limit abuse, etc.)."""
    log_audit(
        event_type=f"security.{event_type}",
        user_id=user_id,
        resource="security",
        success=False,
        details=details,
        ip_address=ip_address,
    )


# ===========================================================================
# Query Helpers (for admin panel / debugging)
# ===========================================================================

def read_recent_file_entries(limit: int = 100) -> list:
    """
    Read the most recent N entries directly from the audit log file.
    Useful as a fallback when DB is unavailable.
    """
    if not AUDIT_LOG_PATH.exists():
        return []

    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]
    except Exception as exc:
        logger.error("Failed to read audit log file: %s", exc)
        return []