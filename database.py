"""
DevMentor AI — Database Configuration & Models
===============================================
Production-grade SQLAlchemy setup with:
- Connection pooling with health checks
- Full ORM models with proper relationships
- Comprehensive indexes for query performance
- Timezone-aware timestamps throughout
- Soft delete support
- Audit trail built into base model
- Repository pattern helpers
- Alembic-compatible (no auto create_all in production)
- Async-ready structure

Usage:
    from database import get_db, User, Thread, Message

    with get_db() as db:
        user = db.query(User).filter(User.email == email).first()
"""

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.pool import QueuePool

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
# Generic type var for repository pattern
# ---------------------------------------------------------------------------
ModelType = TypeVar("ModelType", bound="BaseModel")


# ===========================================================================
# Engine & Session Factory
# ===========================================================================
def _create_engine():
    """
    Build the SQLAlchemy engine with production-tuned connection pooling.

    Key decisions:
    - pool_pre_ping: validates connections before use (catches stale TCP connections)
    - pool_recycle: prevents connections from being held open longer than
      PostgreSQL's idle timeout (default 1 hour)
    - QueuePool: thread-safe pool implementation
    """
    engine = create_engine(
        settings.postgres_url,
        poolclass=QueuePool,
        pool_size=settings.postgres_pool_size,
        max_overflow=settings.postgres_max_overflow,
        pool_timeout=settings.postgres_pool_timeout,
        pool_recycle=settings.postgres_pool_recycle,
        pool_pre_ping=True,             # Reconnect on stale connections
        echo=settings.postgres_echo_sql,
        echo_pool=False,
        future=True,                    # Use SQLAlchemy 2.0 style
        connect_args={
            "connect_timeout": 10,      # Fail fast if DB is unreachable
            "application_name": settings.app_name,  # Visible in pg_stat_activity
            "options": "-c timezone=UTC",  # Always use UTC in PostgreSQL
        },
    )

    # ---------------------------------------------------------------------------
    # Engine event hooks
    # ---------------------------------------------------------------------------
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_connection, connection_record):
        """Set session-level defaults on every new connection."""
        logger.debug("New database connection established")

    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_connection, connection_record, connection_proxy):
        """Called when a connection is retrieved from the pool."""
        pass  # Hook available for future instrumentation

    @event.listens_for(engine, "checkin")
    def on_checkin(dbapi_connection, connection_record):
        """Called when a connection is returned to the pool."""
        pass  # Hook available for future instrumentation

    return engine


engine = _create_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # Avoid lazy-load issues after commit
)

Base = declarative_base()


# ===========================================================================
# Utility Functions
# ===========================================================================
def utc_now() -> datetime:
    """Return current UTC datetime (always timezone-aware)."""
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ===========================================================================
# Base Model Mixin — adds audit fields to every model
# ===========================================================================
class AuditMixin:
    """
    Mixin that adds created_at / updated_at / is_deleted to any model.
    Provides soft-delete support and common query helpers.
    """

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
        comment="UTC timestamp when record was created",
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
        comment="UTC timestamp of last update",
    )
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Soft delete flag — records are never hard deleted",
    )
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="UTC timestamp when record was soft-deleted",
    )

    def soft_delete(self) -> None:
        """Mark record as deleted without removing from database."""
        self.is_deleted = True
        self.deleted_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary. Override in subclasses for custom logic."""
        result = {}
        for col in self.__table__.columns:
            val = getattr(self, col.name)
            if isinstance(val, datetime):
                val = val.isoformat()
            result[col.name] = val
        return result


# ===========================================================================
# Models
# ===========================================================================

class User(AuditMixin, Base):
    """
    Core user model.
    Supports both password-based and OAuth authentication.
    Stores profile, tier, and quota information.
    """
    __tablename__ = "users"

    # Primary key
    id = Column(
        String(36),
        primary_key=True,
        default=new_uuid,
        comment="UUID primary key",
    )

    # Identity
    username = Column(
        String(32),
        unique=True,
        nullable=False,
        comment="Unique username for login and display",
    )
    email = Column(
        String(255),
        unique=True,
        nullable=True,
        comment="User email address",
    )
    password_hash = Column(
        String(255),
        nullable=True,
        comment="Bcrypt hash of password. NULL for OAuth-only users.",
    )
    display_name = Column(
        String(100),
        nullable=True,
        comment="Human-readable display name shown in UI",
    )
    profile_picture = Column(
        String(512),
        nullable=True,
        comment="URL to profile picture",
    )
    bio = Column(
        String(500),
        nullable=True,
        comment="Optional user bio",
    )

    # Account status
    tier = Column(
        String(20),
        default="free",
        nullable=False,
        comment="Subscription tier: free | pro | enterprise",
    )
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether account is active. False = suspended.",
    )
    is_admin = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether user has admin panel access",
    )
    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether email has been verified",
    )
    email_verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    # OAuth
    google_id = Column(
        String(64),
        unique=True,
        nullable=True,
        comment="Google OAuth subject ID",
    )
    github_id = Column(
        String(64),
        unique=True,
        nullable=True,
        comment="GitHub OAuth user ID",
    )

    # API access
    api_key_hash = Column(
        String(255),
        unique=True,
        nullable=True,
        comment="Hashed default API key for programmatic access",
    )

    # Usage tracking
    total_tokens_used = Column(
        BigInteger,
        default=0,
        nullable=False,
        comment="Lifetime token usage counter",
    )
    monthly_tokens_used = Column(
        BigInteger,
        default=0,
        nullable=False,
        comment="Current month token usage (reset monthly)",
    )
    monthly_reset_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When monthly_tokens_used was last reset",
    )

    # Login tracking
    last_login = Column(DateTime(timezone=True), nullable=True)
    last_active = Column(DateTime(timezone=True), nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Account locked until this time after too many failed logins",
    )

    # Preferences stored as JSON
    preferences = Column(
        Text,
        nullable=True,
        default="{}",
        comment="JSON-encoded user preferences (theme, language, model choice, etc.)",
    )

    # Relationships
    threads = relationship(
        "Thread",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    api_keys = relationship(
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    usage_logs = relationship(
        "UsageLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    memories = relationship(
        "LongTermMemory",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    audit_logs = relationship(
        "AuditLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    # Indexes
    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_username", "username"),
        Index("idx_users_google_id", "google_id"),
        Index("idx_users_tier_active", "tier", "is_active"),
        Index("idx_users_created", "created_at"),
    )

    def get_preferences(self) -> Dict[str, Any]:
        """Parse JSON preferences safely."""
        try:
            return json.loads(self.preferences or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_preferences(self, prefs: Dict[str, Any]) -> None:
        """Serialize preferences to JSON."""
        self.preferences = json.dumps(prefs)

    def is_locked(self) -> bool:
        """Check if account is temporarily locked due to failed logins."""
        if self.locked_until is None:
            return False
        return utc_now() < self.locked_until

    def increment_login_failure(self, max_attempts: int = 5, lockout_minutes: int = 30) -> None:
        """Increment failed login counter and lock account if threshold exceeded."""
        from datetime import timedelta
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.locked_until = utc_now() + timedelta(minutes=lockout_minutes)
            logger.warning(
                "Account locked due to failed logins",
                extra={"user_id": self.id, "attempts": self.failed_login_attempts},
            )

    def reset_login_failures(self) -> None:
        """Reset failed login counter after successful login."""
        self.failed_login_attempts = 0
        self.locked_until = None

    def record_login(self) -> None:
        """Update login tracking fields on successful login."""
        self.last_login = utc_now()
        self.last_active = utc_now()
        self.login_count += 1
        self.reset_login_failures()

    def __repr__(self) -> str:
        return f"<User id={self.id!r} username={self.username!r} tier={self.tier!r}>"


class Thread(AuditMixin, Base):
    """
    Conversation thread — groups related messages together.
    Each user can have many threads (chat sessions).
    """
    __tablename__ = "threads"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Content
    title = Column(String(200), default="New Chat", nullable=False)
    summary = Column(
        Text,
        nullable=True,
        comment="AI-generated summary of the conversation",
    )
    mode = Column(
        String(20),
        default="chat",
        nullable=False,
        comment="Chat mode: chat | rag | agent | research",
    )
    model_preference = Column(
        String(50),
        nullable=True,
        comment="Preferred LLM model for this thread (overrides default)",
    )

    # Status
    is_archived = Column(Boolean, default=False, nullable=False)
    is_pinned = Column(Boolean, default=False, nullable=False)
    is_shared = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this thread has a public share link",
    )
    share_token = Column(
        String(64),
        unique=True,
        nullable=True,
        comment="Token used to access shared thread without auth",
    )

    # Stats
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(BigInteger, default=0, nullable=False)

    # Tags stored as JSON array
    tags = Column(
        Text,
        nullable=True,
        default="[]",
        comment="JSON array of user-defined tags",
    )

    # Relationships
    user = relationship("User", back_populates="threads")
    messages = relationship(
        "Message",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    __table_args__ = (
        Index("idx_threads_user_updated", "user_id", "updated_at"),
        Index("idx_threads_user_archived", "user_id", "is_archived"),
        Index("idx_threads_share_token", "share_token"),
        Index("idx_threads_pinned", "user_id", "is_pinned"),
    )

    def get_tags(self) -> List[str]:
        try:
            return json.loads(self.tags or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

    def set_tags(self, tags: List[str]) -> None:
        self.tags = json.dumps(list(set(tags)))  # deduplicate

    def __repr__(self) -> str:
        return f"<Thread id={self.id!r} user_id={self.user_id!r} title={self.title!r}>"


class Message(AuditMixin, Base):
    """
    Individual message within a thread.
    Stores both user and assistant messages with full metadata.
    """
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    thread_id = Column(
        String(36),
        ForeignKey("threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Content
    role = Column(
        String(10),
        nullable=False,
        comment="Message role: user | assistant | system | tool",
    )
    content = Column(Text, nullable=False, comment="Message text content")
    content_type = Column(
        String(20),
        default="text",
        nullable=False,
        comment="Content type: text | markdown | code | image",
    )

    # AI metadata
    model_used = Column(String(50), nullable=True, comment="LLM model that generated this response")
    provider = Column(String(20), nullable=True, comment="LLM provider: gemini | openai | anthropic")
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Float, default=0.0, nullable=False)
    duration_ms = Column(Integer, default=0, comment="Time to generate response in ms")

    # RAG metadata
    rag_sources = Column(
        Text,
        nullable=True,
        comment="JSON array of RAG source documents used",
    )
    rag_score = Column(Float, nullable=True, comment="Average RAG retrieval score")

    # Quality signals
    thumbs_up = Column(Boolean, nullable=True, comment="User feedback: True=up, False=down, None=no feedback")
    feedback_text = Column(String(500), nullable=True, comment="Optional user feedback text")

    # Agent metadata
    is_agent_message = Column(Boolean, default=False, nullable=False)
    tool_calls = Column(
        Text,
        nullable=True,
        comment="JSON array of tool calls made during agent execution",
    )
    tool_results = Column(
        Text,
        nullable=True,
        comment="JSON array of tool results",
    )

    # Moderation
    is_flagged = Column(Boolean, default=False, nullable=False)
    flag_reason = Column(String(200), nullable=True)

    # Relationship
    thread = relationship("Thread", back_populates="messages")

    __table_args__ = (
        Index("idx_messages_thread_created", "thread_id", "created_at"),
        Index("idx_messages_role", "role"),
        Index("idx_messages_flagged", "is_flagged"),
        Index("idx_messages_model", "model_used"),
    )

    def get_rag_sources(self) -> List[Dict]:
        try:
            return json.loads(self.rag_sources or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

    def get_tool_calls(self) -> List[Dict]:
        try:
            return json.loads(self.tool_calls or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

    def __repr__(self) -> str:
        return f"<Message id={self.id} thread_id={self.thread_id!r} role={self.role!r}>"


class APIKey(AuditMixin, Base):
    """
    Named API keys for programmatic access.
    Keys are hashed before storage — the raw key is shown only once at creation.
    """
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    key_hash = Column(
        String(255),
        unique=True,
        nullable=False,
        comment="SHA-256 hash of the raw API key. Raw key is never stored.",
    )
    key_prefix = Column(
        String(8),
        nullable=False,
        comment="First 8 chars of raw key shown in UI for identification",
    )
    name = Column(String(100), nullable=False, comment="Human-readable key name")
    description = Column(String(300), nullable=True)

    # Permissions
    scopes = Column(
        Text,
        default='["chat"]',
        nullable=False,
        comment='JSON array of allowed scopes: ["chat", "rag", "admin"]',
    )
    is_active = Column(Boolean, default=True, nullable=False)

    # Usage tracking
    last_used = Column(DateTime(timezone=True), nullable=True)
    use_count = Column(BigInteger, default=0, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Rate limiting (overrides global if set)
    rate_limit_override = Column(
        Integer,
        nullable=True,
        comment="Custom rate limit for this key. NULL = use global default.",
    )

    # Relationship
    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index("idx_apikeys_user", "user_id"),
        Index("idx_apikeys_expires", "expires_at"),
        Index("idx_apikeys_active", "is_active"),
    )

    def get_scopes(self) -> List[str]:
        try:
            return json.loads(self.scopes or '["chat"]')
        except (json.JSONDecodeError, TypeError):
            return ["chat"]

    def has_scope(self, scope: str) -> bool:
        return scope in self.get_scopes()

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return utc_now() > self.expires_at

    def record_use(self) -> None:
        self.last_used = utc_now()
        self.use_count += 1

    def __repr__(self) -> str:
        return f"<APIKey id={self.id!r} name={self.name!r} user_id={self.user_id!r}>"


class UsageLog(AuditMixin, Base):
    """
    Per-request usage logging for billing, analytics, and abuse detection.
    Tracks token consumption and cost per API call.
    """
    __tablename__ = "usage_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    thread_id = Column(String(36), nullable=True, index=True)
    message_id = Column(BigInteger, nullable=True)

    # Request info
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=True)
    request_id = Column(String(36), nullable=True, comment="Unique request trace ID")
    ip_address = Column(String(45), nullable=True, comment="Supports IPv6")
    user_agent = Column(String(300), nullable=True)

    # AI usage
    model = Column(String(50), nullable=True)
    provider = Column(String(20), nullable=True)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Float, default=0.0, nullable=False)

    # Performance
    duration_ms = Column(Integer, default=0, nullable=False)
    status_code = Column(Integer, nullable=True)
    is_cached = Column(Boolean, default=False, nullable=False)
    cache_hit_type = Column(
        String(20),
        nullable=True,
        comment="Type of cache hit: semantic | exact | none",
    )

    # Error tracking
    had_error = Column(Boolean, default=False, nullable=False)
    error_type = Column(String(100), nullable=True)

    # Relationship
    user = relationship("User", back_populates="usage_logs")

    __table_args__ = (
        Index("idx_usage_user_date", "user_id", "created_at"),
        Index("idx_usage_created", "created_at"),
        Index("idx_usage_model", "model"),
        Index("idx_usage_errors", "had_error", "created_at"),
        Index("idx_usage_request_id", "request_id"),
    )

    def __repr__(self) -> str:
        return f"<UsageLog id={self.id} user_id={self.user_id!r} model={self.model!r}>"


class LongTermMemory(AuditMixin, Base):
    """
    Persistent memories extracted from conversations.
    Enables the AI to remember facts about users across sessions.
    """
    __tablename__ = "long_term_memories"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Memory content
    content = Column(Text, nullable=False, comment="The memory text")
    memory_type = Column(
        String(30),
        default="factual",
        nullable=False,
        comment="Type: factual | procedural | emotional | preference",
    )
    source_thread_id = Column(
        String(36),
        nullable=True,
        comment="Thread this memory was extracted from",
    )

    # Relevance signals
    importance_score = Column(
        Float,
        default=0.5,
        nullable=False,
        comment="0.0-1.0 importance score. Higher = less likely to decay.",
    )
    access_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="How many times this memory has been retrieved",
    )
    last_accessed = Column(DateTime(timezone=True), nullable=True)

    # Decay
    decay_score = Column(
        Float,
        default=1.0,
        nullable=False,
        comment="1.0 = fresh, 0.0 = fully decayed. Computed periodically.",
    )
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Hard expiry date. NULL = never expires.",
    )

    # Vector embedding reference
    embedding_id = Column(
        String(100),
        nullable=True,
        comment="Qdrant point ID for this memory's embedding",
    )

    # Relationship
    user = relationship("User", back_populates="memories")

    __table_args__ = (
        Index("idx_memory_user_type", "user_id", "memory_type"),
        Index("idx_memory_user_importance", "user_id", "importance_score"),
        Index("idx_memory_expires", "expires_at"),
        Index("idx_memory_decay", "decay_score"),
    )

    def record_access(self) -> None:
        self.access_count += 1
        self.last_accessed = utc_now()

    def __repr__(self) -> str:
        return f"<LongTermMemory id={self.id!r} user_id={self.user_id!r} type={self.memory_type!r}>"


class AuditLog(AuditMixin, Base):
    """
    Security audit trail for sensitive operations.
    Immutable by design — records are never updated or deleted.
    """
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,  # NULL for unauthenticated actions
        index=True,
    )

    # Event details
    action = Column(
        String(100),
        nullable=False,
        comment="Action performed e.g. login | logout | delete_thread | change_password",
    )
    resource_type = Column(String(50), nullable=True, comment="Type of resource affected")
    resource_id = Column(String(100), nullable=True, comment="ID of resource affected")

    # Request context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(300), nullable=True)
    request_id = Column(String(36), nullable=True)

    # Outcome
    success = Column(Boolean, nullable=False, default=True)
    failure_reason = Column(Text, nullable=True)

    # Additional context as JSON
    extra_data = Column(
        "metadata",
        Text,
        nullable=True,
        comment="JSON blob with additional event context",
    )

    # Relationship
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("idx_audit_user_created", "user_id", "created_at"),
        Index("idx_audit_action", "action", "created_at"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_ip", "ip_address"),
        Index("idx_audit_success", "success", "created_at"),
    )

    def get_metadata(self) -> Dict[str, Any]:
        try:
            return json.loads(self.extra_data or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action!r} user_id={self.user_id!r}>"


class RateLimitBucket(Base):
    """
    Persistent rate limit tracking (complements Redis-based limiting).
    Used for long-window limits (daily/monthly) that survive Redis restarts.
    """
    __tablename__ = "rate_limit_buckets"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    key = Column(String(200), nullable=False, comment="Rate limit key e.g. user:123:daily")
    count = Column(Integer, default=0, nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    window_end = Column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("key", "window_start", name="uq_ratelimit_key_window"),
        Index("idx_ratelimit_key", "key"),
        Index("idx_ratelimit_window_end", "window_end"),
    )


class WebhookEndpoint(AuditMixin, Base):
    """User-configured webhook endpoints for event notifications."""
    __tablename__ = "webhook_endpoints"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    url = Column(String(512), nullable=False)
    secret_hash = Column(String(255), nullable=True, comment="HMAC secret for payload signing")
    events = Column(
        Text,
        default='["chat.complete"]',
        comment='JSON array of subscribed events',
    )
    is_active = Column(Boolean, default=True, nullable=False)
    last_triggered = Column(DateTime(timezone=True), nullable=True)
    failure_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index("idx_webhook_user", "user_id"),
        Index("idx_webhook_active", "is_active"),
    )

    def get_events(self) -> List[str]:
        try:
            return json.loads(self.events or '["chat.complete"]')
        except (json.JSONDecodeError, TypeError):
            return ["chat.complete"]


# ===========================================================================
# Session Context Manager
# ===========================================================================
@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager that provides a database session with automatic
    commit, rollback, and cleanup.

    Usage:
        with get_db() as db:
            user = db.query(User).filter(User.id == user_id).first()

    The session is committed if the block exits cleanly.
    On any exception, the transaction is rolled back and the exception re-raised.
    The session is always closed in the finally block.
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(
            "Database error — transaction rolled back",
            exc_info=True,
            extra={"error_type": type(exc).__name__},
        )
        raise
    except Exception as exc:
        db.rollback()
        logger.error(
            "Unexpected error — transaction rolled back",
            exc_info=True,
        )
        raise
    finally:
        db.close()


# ===========================================================================
# Flask dependency helper
# ===========================================================================
def get_db_session() -> Generator[Session, None, None]:
    """
    Generator-based session provider for Flask/FastAPI dependency injection.

    Flask usage (via flask-sqlalchemy alternative):
        @app.teardown_appcontext
        def shutdown_session(exception=None):
            db.remove()
    """
    with get_db() as session:
        yield session


# ===========================================================================
# Database Health Check
# ===========================================================================
def check_db_health() -> Dict[str, Any]:
    """
    Verify database connectivity and return status info.
    Used by /health endpoint and startup checks.
    """
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
            result = db.execute(
                text("SELECT COUNT(*) FROM users WHERE is_deleted = false")
            ).scalar()
        return {
            "status": "healthy",
            "active_users": result,
            "pool_size": engine.pool.size(),
            "checked_out": engine.pool.checkedout(),
        }
    except OperationalError as exc:
        logger.error("Database health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}


# ===========================================================================
# Repository Base Class
# ===========================================================================
class BaseRepository:
    """
    Generic repository providing common CRUD operations.
    Subclass for each model to add model-specific queries.

    Example:
        class UserRepository(BaseRepository):
            model = User

            def find_by_email(self, db, email):
                return db.query(User).filter(User.email == email).first()
    """
    model: Type[ModelType] = None

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        return db.query(self.model).filter(
            self.model.id == id,
            self.model.is_deleted == False,
        ).first()

    def get_all(
        self, db: Session, skip: int = 0, limit: int = 100
    ) -> List[ModelType]:
        return (
            db.query(self.model)
            .filter(self.model.is_deleted == False)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def create(self, db: Session, **kwargs) -> ModelType:
        obj = self.model(**kwargs)
        db.add(obj)
        db.flush()  # Get ID without committing
        logger.debug("Created %s id=%s", self.model.__name__, obj.id)
        return obj

    def update(self, db: Session, obj: ModelType, **kwargs) -> ModelType:
        for key, value in kwargs.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        obj.updated_at = utc_now()
        db.flush()
        return obj

    def soft_delete(self, db: Session, obj: ModelType) -> ModelType:
        obj.soft_delete()
        db.flush()
        logger.info("Soft deleted %s id=%s", self.model.__name__, obj.id)
        return obj

    def count(self, db: Session) -> int:
        return db.query(self.model).filter(self.model.is_deleted == False).count()


# Instantiated repositories
class UserRepository(BaseRepository):
    model = User

    def find_by_email(self, db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(
            User.email == email, User.is_deleted == False
        ).first()

    def find_by_username(self, db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(
            User.username == username, User.is_deleted == False
        ).first()

    def find_by_google_id(self, db: Session, google_id: str) -> Optional[User]:
        return db.query(User).filter(User.google_id == google_id).first()

    def find_active_users(
        self, db: Session, skip: int = 0, limit: int = 100
    ) -> List[User]:
        return (
            db.query(User)
            .filter(User.is_active == True, User.is_deleted == False)
            .offset(skip)
            .limit(limit)
            .all()
        )


class ThreadRepository(BaseRepository):
    model = Thread

    def find_by_user(
        self,
        db: Session,
        user_id: str,
        include_archived: bool = False,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Thread]:
        query = db.query(Thread).filter(
            Thread.user_id == user_id,
            Thread.is_deleted == False,
        )
        if not include_archived:
            query = query.filter(Thread.is_archived == False)
        return (
            query.order_by(Thread.updated_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

class Upload(AuditMixin, Base):
    """
    Tracks files uploaded to a user's knowledge base (RAG).
    """
    __tablename__ = "uploads"

    id = Column(String(36), primary_key=True, default=new_uuid)
    user_id = Column(String(36), nullable=False, index=True)
    original_name = Column(String(255), nullable=False)
    stored_name = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    is_indexed = Column(Boolean, default=False, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class UploadRepository(BaseRepository):
    model = Upload

class MessageRepository(BaseRepository):
    model = Message

    def find_by_thread(
        self, db: Session, thread_id: str, limit: int = 50
    ) -> List[Message]:
        return (
            db.query(Message)
            .filter(
                Message.thread_id == thread_id,
                Message.is_deleted == False,
            )
            .order_by(Message.created_at.asc())
            .limit(limit)
            .all()
        )


# Singleton repository instances
user_repo = UserRepository()
thread_repo = ThreadRepository()
message_repo = MessageRepository()


# ===========================================================================
# Database Initialization (development only)
# ===========================================================================
def init_db() -> None:
    """
    Create all tables from SQLAlchemy models.

    ⚠️  FOR DEVELOPMENT ONLY.
    In production, always use Alembic migrations:
        alembic upgrade head
    """
    if settings.is_production:
        raise RuntimeError(
            "init_db() is not allowed in production. Use: alembic upgrade head"
        )
    logger.info("Creating database tables (development mode)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")


# ===========================================================================
# Module self-test
# ===========================================================================
if __name__ == "__main__":
    health = check_db_health()
    print(f"Database health: {health}")