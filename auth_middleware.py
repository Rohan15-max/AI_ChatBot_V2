"""
DevMentor AI — Authentication Middleware
=========================================
Production-grade authentication layer supporting:

- JWT access tokens with issuer/audience/JTI validation
- JWT refresh tokens with rotation
- API key authentication (SHA-256 indexed + bcrypt verified)
- Google OAuth token verification
- Session-based authentication (Flask sessions)
- Account lockout after failed attempts
- JWT blacklist check (Redis-backed)
- Role-based access control (RBAC) decorators
- Scope-based API key authorization
- Structured security logging
- Request context enrichment

Usage:
    from auth_middleware import require_auth, require_admin, require_scope

    @app.route("/chat", methods=["POST"])
    @require_auth
    def chat():
        user_id = g.user_id
"""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import bcrypt
import jwt
from flask import g, jsonify, request, session

from config import get_settings
from database import APIKey, AuditLog, User, get_db, utc_now

logger = logging.getLogger(__name__)
settings = get_settings()

JWT_ISSUER = "devmentor.ai"
JWT_AUDIENCE = "devmentor-api"
JWT_REFRESH_AUDIENCE = "devmentor-refresh"
BEARER_PREFIX = "bearer"
API_KEY_HEADER = "X-API-Key"
MIN_API_KEY_LENGTH = 32


# ===========================================================================
# Token Generation
# ===========================================================================

def generate_access_token(
    user_id: str,
    username: str,
    tier: str = "free",
    is_admin: bool = False,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a signed JWT access token with full claims."""
    now = utc_now()
    jti = str(uuid.uuid4())

    payload: Dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "tier": tier,
        "is_admin": is_admin,
        "jti": jti,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expiry_hours),
        "type": "access",
    }

    if extra_claims:
        protected = {"sub", "jti", "iss", "aud", "iat", "exp", "type"}
        safe_extras = {k: v for k, v in extra_claims.items() if k not in protected}
        payload.update(safe_extras)

    return jwt.encode(
        payload,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def generate_refresh_token(user_id: str) -> Tuple[str, str]:
    """
    Generate a JWT refresh token.
    Returns: Tuple of (token_string, jti)
    """
    now = utc_now()
    jti = str(uuid.uuid4())

    payload = {
        "sub": user_id,
        "jti": jti,
        "iss": JWT_ISSUER,
        "aud": JWT_REFRESH_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(days=settings.jwt_refresh_expiry_days),
        "type": "refresh",
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token, jti


def generate_token_pair(
    user_id: str,
    username: str,
    tier: str = "free",
    is_admin: bool = False,
) -> Dict[str, Any]:
    """Generate both access and refresh tokens."""
    access_token = generate_access_token(user_id, username, tier, is_admin)
    refresh_token, _ = generate_refresh_token(user_id)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": settings.jwt_expiry_hours * 3600,
    }


# ===========================================================================
# Token Decoding & Validation
# ===========================================================================

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and fully validate a JWT access token.
    Checks: signature, expiry, issuer, audience, type, JTI blacklist.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
            options={"require": ["exp", "iat", "sub", "jti", "type"]},
        )

        if payload.get("type") != "access":
            logger.warning("Token type mismatch — expected 'access'")
            return None

        jti = payload.get("jti")
        if jti and _is_token_blacklisted(jti):
            logger.warning("Blacklisted token used", extra={"jti": jti})
            return None

        return payload

    except jwt.ExpiredSignatureError:
        logger.info("Expired JWT presented")
        return None
    except jwt.InvalidAudienceError:
        logger.warning("JWT audience mismatch")
        return None
    except jwt.InvalidIssuerError:
        logger.warning("JWT issuer mismatch")
        return None
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid JWT: %s", exc)
        return None


def decode_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT refresh token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            audience=JWT_REFRESH_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        if payload.get("type") != "refresh":
            return None
        jti = payload.get("jti")
        if jti and _is_token_blacklisted(jti):
            return None
        return payload
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid refresh token: %s", exc)
        return None


def blacklist_token(jti: str, expires_in_seconds: int = 86400) -> bool:
    """Add a JWT JTI to the Redis blacklist."""
    try:
        from security.jwt_blacklist import add_to_blacklist
        return add_to_blacklist(jti, expires_in_seconds)
    except Exception as exc:
        logger.error("Failed to blacklist JTI %s: %s", jti, exc)
        return False


def _is_token_blacklisted(jti: str) -> bool:
    """Check Redis blacklist for a JTI."""
    try:
        from security.jwt_blacklist import is_blacklisted
        return is_blacklisted(jti)
    except Exception:
        return False  # Fail open if Redis is down


# ===========================================================================
# Password & API Key Helpers
# ===========================================================================

def hash_password(plain_password: str) -> str:
    """Hash a password with bcrypt."""
    salt = bcrypt.gensalt(rounds=settings.bcrypt_rounds)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed: str) -> bool:
    """Verify a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as exc:
        logger.error("Password verification error: %s", exc)
        return False


def generate_api_key() -> Tuple[str, str, str]:
    """
    Generate a new API key with dual hashing.
    - SHA-256: fast DB lookup
    - Bcrypt: secure verification

    Returns: (raw_key, sha256_hash, bcrypt_hash)
    Show raw_key to user ONCE — never store it.
    """
    raw_key = f"dm_{secrets.token_urlsafe(48)}"
    sha256_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    bcrypt_hash = bcrypt.hashpw(
        raw_key.encode("utf-8"),
        bcrypt.gensalt(rounds=settings.bcrypt_rounds),
    ).decode("utf-8")
    return raw_key, sha256_hash, bcrypt_hash


def verify_api_key_value(raw_key: str, bcrypt_hash: str) -> bool:
    """Verify a raw API key against its bcrypt hash."""
    try:
        return bcrypt.checkpw(raw_key.encode("utf-8"), bcrypt_hash.encode("utf-8"))
    except Exception:
        return False


def sha256_api_key(raw_key: str) -> str:
    """Compute SHA-256 of API key for fast DB lookup."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ===========================================================================
# Request Context Helpers
# ===========================================================================

def _enrich_request_context(user: User) -> None:
    """Attach user info to Flask g and request objects."""
    g.user_id = user.id
    g.username = user.username
    g.user_tier = user.tier
    g.is_admin = user.is_admin
    g.user = user
    request.user_id = user.id
    request.username = user.username
    request.user_tier = user.tier
    request.is_admin = user.is_admin


def get_current_user_id() -> Optional[str]:
    return getattr(g, "user_id", None)


def get_current_user() -> Optional[User]:
    return getattr(g, "user", None)


def _get_client_ip() -> str:
    """Extract real client IP respecting X-Forwarded-For."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _extract_bearer_token() -> Optional[str]:
    """Extract Bearer token from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    parts = auth_header.split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() == BEARER_PREFIX:
        return parts[1].strip()
    return None


def _log_auth_failure(reason: str, user_id: Optional[str] = None, extra: Optional[Dict] = None) -> None:
    """Log authentication failure with context."""
    log_extra = {
        "ip": _get_client_ip(),
        "path": request.path,
        "method": request.method,
        "user_agent": request.headers.get("User-Agent", "")[:200],
        "reason": reason,
    }
    if user_id:
        log_extra["user_id"] = user_id
    if extra:
        log_extra.update(extra)
    logger.warning("Auth failure: %s", reason, extra=log_extra)


def _write_audit_log(
    action: str,
    user_id: Optional[str],
    success: bool,
    failure_reason: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
) -> None:
    """Write security audit log entry — never raises."""
    try:
        with get_db() as db:
            entry = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=_get_client_ip(),
                user_agent=request.headers.get("User-Agent", "")[:300],
                request_id=getattr(g, "request_id", None),
                success=success,
                failure_reason=failure_reason,
            )
            db.add(entry)
    except Exception as exc:
        logger.error("Audit log write failed: %s", exc)


# ===========================================================================
# Auth Resolution
# ===========================================================================

def _resolve_user_from_request() -> Tuple[Optional[User], Optional[str]]:
    """
    Try JWT → Session → API Key in order.
    Returns: (User or None, failure_reason or None)
    """
    # ---- JWT Bearer ----
    token = _extract_bearer_token()
    if token:
        payload = decode_access_token(token)
        if not payload:
            return None, "invalid_or_expired_jwt"
        user_id = payload.get("sub")
        if not user_id:
            return None, "missing_sub_claim"
        with get_db() as db:
            user = db.query(User).filter(
                User.id == user_id, User.is_deleted == False
            ).first()
            if not user:
                return None, "user_not_found"
            if not user.is_active:
                return None, "account_inactive"
            if user.is_locked():
                return None, "account_locked"
            user.last_active = utc_now()
        return user, None

    # ---- Flask Session ----
    session_user_id = session.get("user_id")
    if session_user_id:
        with get_db() as db:
            user = db.query(User).filter(
                User.id == session_user_id, User.is_deleted == False
            ).first()
            if not user:
                session.clear()
                return None, "session_user_not_found"
            if not user.is_active:
                session.clear()
                return None, "account_inactive"
            if user.is_locked():
                return None, "account_locked"
            user.last_active = utc_now()
        return user, None

    # ---- API Key ----
    api_key_value = request.headers.get(API_KEY_HEADER)
    if api_key_value:
        if len(api_key_value) < MIN_API_KEY_LENGTH:
            return None, "api_key_too_short"
        key_sha256 = sha256_api_key(api_key_value)
        with get_db() as db:
            api_key_record = db.query(APIKey).filter(
                APIKey.key_hash == key_sha256,
                APIKey.is_deleted == False,
                APIKey.is_active == True,
            ).first()
            if not api_key_record:
                return None, "api_key_not_found"
            if api_key_record.is_expired():
                return None, "api_key_expired"
            if not verify_api_key_value(api_key_value, api_key_record.key_hash):
                return None, "api_key_hash_mismatch"
            user = db.query(User).filter(
                User.id == api_key_record.user_id, User.is_deleted == False
            ).first()
            if not user or not user.is_active:
                return None, "api_key_user_inactive"
            api_key_record.record_use()
            g.api_key_scopes = api_key_record.get_scopes()
            g.api_key_id = api_key_record.id
        return user, None

    return None, "no_credentials"


# ===========================================================================
# Decorators
# ===========================================================================

def require_auth(f: Callable) -> Callable:
    """Require any valid authentication (JWT, session, or API key)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user, failure_reason = _resolve_user_from_request()
        if user is None:
            _log_auth_failure(failure_reason or "unknown")
            _write_audit_log("auth.failed", None, False, failure_reason)
            return _auth_error_response(failure_reason)
        _enrich_request_context(user)
        return f(*args, **kwargs)
    return decorated


def require_admin(f: Callable) -> Callable:
    """Require authentication AND admin privileges."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user, failure_reason = _resolve_user_from_request()
        if user is None:
            _log_auth_failure(failure_reason or "unknown")
            return _auth_error_response(failure_reason)
        if not user.is_admin:
            _log_auth_failure("admin_required", user_id=user.id)
            _write_audit_log("admin.access_denied", user.id, False, "not_admin")
            return jsonify({"error": "Admin privileges required", "code": "FORBIDDEN"}), 403
        _enrich_request_context(user)
        _write_audit_log("admin.access", user.id, True, resource_type="endpoint", resource_id=request.path)
        return f(*args, **kwargs)
    return decorated


def require_tier(*allowed_tiers: str) -> Callable:
    """Restrict access to specific subscription tiers."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            user, failure_reason = _resolve_user_from_request()
            if user is None:
                return _auth_error_response(failure_reason)
            if user.tier not in allowed_tiers:
                _log_auth_failure("insufficient_tier", user_id=user.id)
                return jsonify({
                    "error": f"Requires tier: {', '.join(allowed_tiers)}",
                    "code": "UPGRADE_REQUIRED",
                    "current_tier": user.tier,
                    "required_tiers": list(allowed_tiers),
                }), 403
            _enrich_request_context(user)
            return f(*args, **kwargs)
        return decorated
    return decorator


def require_scope(*required_scopes: str) -> Callable:
    """Validate API key scopes. JWT/session auth bypasses scope checks."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            api_key_scopes = getattr(g, "api_key_scopes", None)
            if api_key_scopes is not None:
                missing = [s for s in required_scopes if s not in api_key_scopes]
                if missing:
                    _log_auth_failure("missing_scope", user_id=getattr(g, "user_id", None))
                    return jsonify({
                        "error": f"Missing scopes: {', '.join(missing)}",
                        "code": "INSUFFICIENT_SCOPE",
                    }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


def optional_auth(f: Callable) -> Callable:
    """Attempt auth but don't require it. Sets g.user_id = None if unauthenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            user, _ = _resolve_user_from_request()
            if user:
                _enrich_request_context(user)
            else:
                g.user_id = None
                g.username = None
                g.user_tier = "anonymous"
                g.is_admin = False
                g.user = None
        except Exception:
            g.user_id = None
            g.user = None
        return f(*args, **kwargs)
    return decorated


# ===========================================================================
# Error Response Builder
# ===========================================================================

def _auth_error_response(reason: Optional[str]) -> tuple:
    """Build consistent 401 error response."""
    messages = {
        "no_credentials":         ("No authentication credentials provided.", "NO_CREDENTIALS"),
        "invalid_or_expired_jwt": ("Token is invalid or has expired.", "TOKEN_INVALID"),
        "missing_sub_claim":      ("Token payload is malformed.", "TOKEN_MALFORMED"),
        "user_not_found":         ("User account not found.", "USER_NOT_FOUND"),
        "account_inactive":       ("Account has been deactivated.", "ACCOUNT_INACTIVE"),
        "account_locked":         ("Account is temporarily locked.", "ACCOUNT_LOCKED"),
        "session_user_not_found": ("Session is invalid. Please log in again.", "SESSION_INVALID"),
        "api_key_not_found":      ("Invalid API key.", "API_KEY_INVALID"),
        "api_key_expired":        ("API key has expired.", "API_KEY_EXPIRED"),
        "api_key_hash_mismatch":  ("Invalid API key.", "API_KEY_INVALID"),
        "api_key_user_inactive":  ("API key owner account is inactive.", "ACCOUNT_INACTIVE"),
    }
    message, code = messages.get(reason or "", ("Authentication required.", "AUTH_REQUIRED"))
    return jsonify({"error": message, "code": code}), 401


# ===========================================================================
# Google OAuth
# ===========================================================================

def verify_google_token(id_token: str) -> Optional[Dict[str, Any]]:
    """Verify a Google ID token and return user info."""
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token

        if not settings.google_client_id:
            logger.error("Google OAuth not configured")
            return None

        id_info = google_id_token.verify_oauth2_token(
            id_token,
            google_requests.Request(),
            settings.google_client_id,
            clock_skew_in_seconds=10,
        )

        if id_info.get("aud") != settings.google_client_id:
            logger.warning("Google token audience mismatch")
            return None

        return {
            "google_id": id_info["sub"],
            "email": id_info.get("email"),
            "email_verified": id_info.get("email_verified", False),
            "display_name": id_info.get("name"),
            "profile_picture": id_info.get("picture"),
        }
    except ImportError:
        logger.error("google-auth not installed. Run: pip install google-auth")
        return None
    except Exception as exc:
        logger.warning("Google token verification failed: %s", exc)
        return None


# ===========================================================================
# Session Management
# ===========================================================================

def create_user_session(user: User) -> None:
    """Create a Flask session after successful login."""
    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username
    session["tier"] = user.tier
    session["is_admin"] = user.is_admin
    session["login_time"] = utc_now().isoformat()
    session.permanent = True
    logger.info("Session created", extra={"user_id": user.id, "ip": _get_client_ip()})


def destroy_user_session() -> Optional[str]:
    """Destroy session and blacklist current JWT if present."""
    user_id = session.get("user_id")
    session.clear()

    token = _extract_bearer_token()
    if token:
        payload = decode_access_token(token)
        if payload:
            jti = payload.get("jti")
            if jti:
                blacklist_token(jti, settings.jwt_expiry_hours * 3600)
                return jti

    if user_id:
        logger.info("Session destroyed", extra={"user_id": user_id})
    return None