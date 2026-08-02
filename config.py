"""
DevMentor AI — Centralized Configuration Management
=====================================================
Single source of truth for all application settings.
Validates, parses, and exposes all environment variables
with strong typing, sensible defaults, and fail-fast
validation on startup so misconfigured deploys are caught
immediately rather than at runtime.

Usage:
    from config import get_settings
    settings = get_settings()
    print(settings.postgres_url)
"""

import logging
import os
import secrets
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set
from urllib.parse import quote_plus

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    validator,
)
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Module-level logger (does NOT use app settings — used during boot only)
# ---------------------------------------------------------------------------
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment type alias
# ---------------------------------------------------------------------------
Environment = Literal["development", "staging", "production", "testing"]


# ===========================================================================
# Main Settings Class
# ===========================================================================
class Settings(BaseSettings):
    """
    All application settings loaded from environment variables / .env file.

    Pydantic validates every field on instantiation. If a required secret
    is missing the process exits with a clear error message — no silent
    failures in production.
    """

    # -----------------------------------------------------------------------
    # APPLICATION
    # -----------------------------------------------------------------------
    app_name: str = Field(
        default="DevMentor AI",
        description="Human-readable application name used in logs and UI.",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Semantic version string shown in /health and API docs.",
    )
    app_env: Environment = Field(
        default="development",
        description="Deployment environment. Controls debug mode, log verbosity, etc.",
    )
    app_secret: SecretStr = Field(
        ...,
        description="Flask/session secret key. Must be at least 32 random bytes.",
    )
    app_host: str = Field(default="0.0.0.0", description="Host the server binds to.")
    app_port: int = Field(default=8000, ge=1, le=65535, description="HTTP port.")
    app_workers: int = Field(
        default=4, ge=1, le=64, description="Number of Gunicorn worker processes."
    )
    debug: bool = Field(
        default=False,
        description="Enable Flask debug mode. NEVER true in production.",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins. In production, set to your actual domain(s).",
    )
    base_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Public base URL of the app e.g. https://devmentor.ai",
    )

    # -----------------------------------------------------------------------
    # DATABASE — PostgreSQL
    # -----------------------------------------------------------------------
    postgres_host: str = Field(default="localhost", description="PostgreSQL hostname.")
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = Field(default="devmentor", description="Database name.")
    postgres_user: str = Field(..., description="PostgreSQL username.")
    postgres_password: SecretStr = Field(..., description="PostgreSQL password.")
    postgres_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="SQLAlchemy connection pool size.",
    )
    postgres_max_overflow: int = Field(
        default=20,
        ge=0,
        le=200,
        description="Max connections above pool_size allowed to overflow.",
    )
    postgres_pool_timeout: int = Field(
        default=30, description="Seconds to wait for a pool connection."
    )
    postgres_pool_recycle: int = Field(
        default=1800,
        description="Recycle connections after this many seconds (prevents stale connections).",
    )
    postgres_echo_sql: bool = Field(
        default=False,
        description="Log every SQL statement. Use only in development.",
    )

    @property
    def postgres_url(self) -> str:
        """Synchronous SQLAlchemy connection URL."""
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_postgres_url(self) -> str:
        """Async SQLAlchemy connection URL (asyncpg driver)."""
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -----------------------------------------------------------------------
    # CACHE & MESSAGE BROKER — Redis
    # -----------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Primary Redis URL for caching and sessions.",
    )
    redis_password: Optional[SecretStr] = Field(
        default=None, description="Redis AUTH password (if required)."
    )
    redis_max_connections: int = Field(
        default=50, ge=1, description="Max Redis connection pool size."
    )
    redis_socket_timeout: int = Field(
        default=5, description="Redis socket timeout in seconds."
    )
    redis_retry_on_timeout: bool = Field(
        default=True, description="Automatically retry timed-out Redis commands."
    )

    # Celery uses separate Redis databases to avoid key collisions
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1",
        description="Celery message broker URL.",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL.",
    )
    celery_task_serializer: str = Field(default="json")
    celery_result_serializer: str = Field(default="json")
    celery_task_time_limit: int = Field(
        default=300, description="Hard time limit per Celery task in seconds."
    )
    celery_task_soft_time_limit: int = Field(
        default=240,
        description="Soft time limit — task receives SoftTimeLimitExceeded before hard kill.",
    )
    celery_worker_concurrency: int = Field(
        default=4, description="Number of concurrent Celery worker processes."
    )

    # -----------------------------------------------------------------------
    # VECTOR DATABASE — Qdrant
    # -----------------------------------------------------------------------
    qdrant_host: str = Field(default="localhost", description="Qdrant server hostname.")
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_grpc_port: int = Field(
        default=6334, description="Qdrant gRPC port for high-throughput operations."
    )
    qdrant_api_key: Optional[SecretStr] = Field(
        default=None, description="Qdrant API key (required for Qdrant Cloud)."
    )
    qdrant_use_grpc: bool = Field(
        default=False,
        description="Use gRPC instead of HTTP for Qdrant (faster for bulk ops).",
    )
    qdrant_collection_name: str = Field(
        default="devmentor_knowledge",
        description="Default Qdrant collection name for RAG documents.",
    )
    qdrant_vector_size: int = Field(
        default=768,
        description="Embedding vector dimensions. Must match your embedding model.",
    )
    qdrant_timeout: int = Field(
        default=30, description="Qdrant client request timeout in seconds."
    )

    # -----------------------------------------------------------------------
    # LLM PROVIDERS
    # -----------------------------------------------------------------------
    # Google Gemini
    gemini_api_key: Optional[SecretStr] = Field(
        default=None, description="Google Gemini API key."
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        description="Default Gemini model name.",
    )
    gemini_pro_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini Pro model for complex tasks.",
    )

    # OpenAI
    openai_api_key: Optional[SecretStr] = Field(
        default=None, description="OpenAI API key."
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model.",
    )
    openai_pro_model: str = Field(
        default="gpt-4o",
        description="OpenAI Pro model for complex tasks.",
    )
    openai_org_id: Optional[str] = Field(
        default=None, description="OpenAI organization ID (optional)."
    )# -----------------------------------------------------------------------
    # WEB SEARCH (Serper — used by agent_tools.py's web_search tool)
    # -----------------------------------------------------------------------
    serper_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Serper.dev API key for real web search results in agent mode. "
                     "Falls back to DuckDuckGo's limited instant-answer API if unset.",
    )
    gemini_embed_model: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model used for RAG and semantic cache vector generation.",
    )

    # Anthropic
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None, description="Anthropic API key."
    )
    anthropic_model: str = Field(
        default="claude-3-haiku-20240307",
        description="Default Anthropic model.",
    )
    anthropic_pro_model: str = Field(
        default="claude-sonnet-4-6",
        description="Anthropic Pro model for complex tasks.",
    )

    # Local / Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL for local model serving.",
    )
    ollama_model: str = Field(
        default="llama3",
        description="Default Ollama model name.",
    )
    ollama_enabled: bool = Field(
        default=False,
        description="Enable Ollama as a fallback LLM provider.",
    )

    # LLM General Settings
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens in LLM response.",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature. Lower = more deterministic.",
    )
    llm_timeout: int = Field(
        default=60, description="LLM API request timeout in seconds."
    )
    llm_max_retries: int = Field(
        default=3, description="Max retry attempts on LLM API failures."
    )
    llm_retry_delay: float = Field(
        default=1.0, description="Base delay in seconds between LLM retries."
    )

    # Cost routing thresholds
    cost_router_enabled: bool = Field(
        default=True,
        description="Enable automatic routing to cheaper models for simple queries.",
    )
    cost_router_complexity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Complexity score above which expensive models are used.",
    )

    # -----------------------------------------------------------------------
    # AUTHENTICATION & SECURITY
    # -----------------------------------------------------------------------
    jwt_secret: SecretStr = Field(
        ..., description="Secret key for signing JWT tokens. Must be strong and random."
    )
    jwt_algorithm: str = Field(
        default="HS256", description="JWT signing algorithm."
    )
    jwt_expiry_hours: int = Field(
        default=24, ge=1, le=720, description="JWT token validity in hours."
    )
    jwt_refresh_expiry_days: int = Field(
        default=30, description="JWT refresh token validity in days."
    )

    # Google OAuth
    google_client_id: Optional[str] = Field(
        default=None, description="Google OAuth 2.0 client ID."
    )
    google_client_secret: Optional[SecretStr] = Field(
        default=None, description="Google OAuth 2.0 client secret."
    )

    # Password hashing
    bcrypt_rounds: int = Field(
        default=12,
        ge=4,
        le=31,
        description="Bcrypt work factor. Higher = slower but more secure.",
    )

    # Session
    session_cookie_secure: bool = Field(
        default=True,
        description="Set Secure flag on session cookies. Disable only in local dev.",
    )
    session_cookie_httponly: bool = Field(
        default=True, description="Prevent JavaScript access to session cookies."
    )
    session_cookie_samesite: str = Field(
        default="Lax", description="SameSite cookie policy: Strict | Lax | None"
    )
    session_lifetime_hours: int = Field(
        default=24, description="Session lifetime in hours."
    )

    # -----------------------------------------------------------------------
    # RATE LIMITING
    # -----------------------------------------------------------------------
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_per_user: int = Field(
        default=100,
        ge=1,
        description="Max requests per user per window.",
    )
    rate_limit_per_ip: int = Field(
        default=200,
        ge=1,
        description="Max requests per IP per window.",
    )
    rate_limit_window: int = Field(
        default=60, ge=1, description="Rate limit window size in seconds."
    )
    rate_limit_burst_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        description="Allow this multiplier of the limit as a burst before throttling.",
    )
    # Endpoints that bypass rate limiting entirely
    rate_limit_exempt_paths: Set[str] = Field(
        default={"/health", "/metrics", "/favicon.ico"},
        description="URL paths exempt from rate limiting.",
    )

    # -----------------------------------------------------------------------
    # RAG — Retrieval Augmented Generation
    # -----------------------------------------------------------------------
    rag_enabled: bool = Field(default=True)
    rag_chunk_size: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Token chunk size for document splitting.",
    )
    rag_chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Overlap tokens between adjacent chunks.",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top chunks to retrieve per query.",
    )
    rag_score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include a chunk in results.",
    )
    rag_reranker_enabled: bool = Field(
        default=True, description="Enable cross-encoder reranking of retrieved chunks."
    )
    rag_reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model for reranking.",
    )
    rag_bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of BM25 scores in hybrid search (rest goes to vector).",
    )
    rag_storage_path: Path = Field(
        default=Path("backend/rag_storage"),
        description="Local filesystem path for RAG document storage.",
    )
    rag_max_file_size_mb: int = Field(
        default=50, ge=1, description="Maximum upload file size in MB."
    )
    rag_allowed_extensions: Set[str] = Field(
        default={".pdf", ".txt", ".md", ".docx", ".csv", ".html"},
        description="Allowed file extensions for RAG document upload.",
    )

    # -----------------------------------------------------------------------
    # MEMORY
    # -----------------------------------------------------------------------
    memory_enabled: bool = Field(default=True)
    memory_max_short_term: int = Field(
        default=20,
        description="Max messages to keep in short-term (in-context) memory.",
    )
    memory_max_long_term: int = Field(
        default=1000,
        description="Max memories to store per user in long-term storage.",
    )
    memory_decay_days: int = Field(
        default=90,
        description="Days after which unused memories start to decay/expire.",
    )
    memory_summarize_threshold: int = Field(
        default=10,
        description="Summarize conversation after this many messages to save context.",
    )

    # -----------------------------------------------------------------------
    # SEMANTIC CACHE
    # -----------------------------------------------------------------------
    semantic_cache_enabled: bool = Field(
        default=True,
        description="Enable semantic caching to avoid redundant LLM calls.",
    )
    semantic_cache_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold to consider a cache hit.",
    )
    semantic_cache_ttl_seconds: int = Field(
        default=3600, description="Cache entry TTL in seconds."
    )
    semantic_cache_max_size: int = Field(
        default=10000, description="Maximum number of entries in semantic cache."
    )

    # -----------------------------------------------------------------------
    # AGENT
    # -----------------------------------------------------------------------
    agent_enabled: bool = Field(default=True)
    agent_max_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum tool-calling iterations per agent run.",
    )
    agent_timeout: int = Field(
        default=120, description="Total agent run timeout in seconds."
    )
    agent_max_tool_retries: int = Field(
        default=2, description="Max retries per tool call on failure."
    )

    # -----------------------------------------------------------------------
    # STREAMING
    # -----------------------------------------------------------------------
    streaming_enabled: bool = Field(
        default=True, description="Enable WebSocket token streaming."
    )
    streaming_chunk_delay_ms: int = Field(
        default=0,
        ge=0,
        description="Artificial delay between streamed chunks in ms (0 = no delay).",
    )
    websocket_ping_interval: int = Field(
        default=25, description="WebSocket ping interval in seconds."
    )
    websocket_ping_timeout: int = Field(
        default=60, description="WebSocket ping timeout in seconds."
    )

    # -----------------------------------------------------------------------
    # MONITORING & OBSERVABILITY
    # -----------------------------------------------------------------------
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(
        default=9090, ge=1, le=65535, description="Prometheus metrics server port."
    )
    sentry_dsn: Optional[str] = Field(
        default=None, description="Sentry DSN for error tracking."
    )
    sentry_traces_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of transactions to send to Sentry for performance monitoring.",
    )
    sentry_environment: Optional[str] = Field(
        default=None,
        description="Sentry environment tag. Defaults to app_env if not set.",
    )

    # -----------------------------------------------------------------------
    # ADMIN PANEL
    # -----------------------------------------------------------------------
    admin_enabled: bool = Field(default=True)
    admin_secret_key: Optional[SecretStr] = Field(
        default=None,
        description="Secret key to access the admin panel. Defaults to app_secret.",
    )
    admin_allowed_ips: Optional[List[str]] = Field(
        default=None,
        description="If set, restrict admin panel access to these IPs only.",
    )

    # -----------------------------------------------------------------------
    # BACKGROUND TASKS
    # -----------------------------------------------------------------------
    background_tasks_enabled: bool = Field(default=True)
    task_max_retries: int = Field(
        default=3, description="Default max retries for background tasks."
    )
    task_retry_backoff: float = Field(
        default=2.0, description="Exponential backoff multiplier for task retries."
    )

    # -----------------------------------------------------------------------
    # FEATURE FLAGS
    # -----------------------------------------------------------------------
    feature_streaming: bool = Field(default=True)
    feature_rag: bool = Field(default=True)
    feature_agent: bool = Field(default=True)
    feature_long_term_memory: bool = Field(default=True)
    feature_semantic_cache: bool = Field(default=True)
    feature_google_oauth: bool = Field(default=False)
    feature_admin_panel: bool = Field(default=True)
    feature_webhooks: bool = Field(default=True)
    feature_analytics: bool = Field(default=True)
    feature_multimodal: bool = Field(default=False)

    # -----------------------------------------------------------------------
    # FILE STORAGE
    # -----------------------------------------------------------------------
    upload_dir: Path = Field(
        default=Path("uploads"),
        description="Directory for temporary file uploads.",
    )
    max_upload_size_mb: int = Field(
        default=50, ge=1, description="Maximum upload size in megabytes."
    )

    # -----------------------------------------------------------------------
    # ANALYTICS
    # -----------------------------------------------------------------------
    analytics_db_path: Path = Field(
        default=Path("analytics.db"),
        description="SQLite path for analytics data.",
    )
    analytics_retention_days: int = Field(
        default=90, description="Days to retain analytics data."
    )

    # -----------------------------------------------------------------------
    # VALIDATORS
    # -----------------------------------------------------------------------
    @validator("app_env", pre=True)
    def normalize_env(cls, v: str) -> str:
        return v.lower().strip()

    @validator("log_level", pre=True)
    def normalize_log_level(cls, v: str) -> str:
        v = v.upper().strip()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return v

    @validator("debug", always=True)
    def no_debug_in_production(cls, v: bool, values: Dict[str, Any]) -> bool:
        if v and values.get("app_env") == "production":
            raise ValueError(
                "debug=True is not allowed in production. "
                "Set APP_ENV=development or disable DEBUG."
            )
        return v

    @validator("session_cookie_secure", always=True)
    def warn_insecure_cookies(cls, v: bool, values: Dict[str, Any]) -> bool:
        if not v and values.get("app_env") == "production":
            _log.warning(
                "session_cookie_secure=False in production is a security risk. "
                "Enable HTTPS and set SESSION_COOKIE_SECURE=true."
            )
        return v

    @validator("gemini_api_key", "openai_api_key", "anthropic_api_key", always=True)
    def at_least_one_llm_key(
        cls, v: Optional[SecretStr], values: Dict[str, Any]
    ) -> Optional[SecretStr]:
        """Ensure at least one LLM provider is configured."""
        # This runs for each field — we check after the last one
        return v

    @validator("rag_chunk_overlap", always=True)
    def overlap_less_than_chunk(cls, v: int, values: Dict[str, Any]) -> int:
        chunk_size = values.get("rag_chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"rag_chunk_overlap ({v}) must be less than rag_chunk_size ({chunk_size})"
            )
        return v

    @validator("rag_storage_path", "upload_dir", always=True, pre=False)
    def ensure_directories_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    # -----------------------------------------------------------------------
    # COMPUTED PROPERTIES
    # -----------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "testing"

    @property
    def effective_sentry_environment(self) -> str:
        return self.sentry_environment or self.app_env

    @property
    def effective_admin_secret(self) -> str:
        if self.admin_secret_key:
            return self.admin_secret_key.get_secret_value()
        return self.app_secret.get_secret_value()

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def active_llm_providers(self) -> List[str]:
        """Returns list of configured LLM providers."""
        providers = []
        if self.gemini_api_key:
            providers.append("gemini")
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.ollama_enabled:
            providers.append("ollama")
        return providers

    def as_safe_dict(self) -> Dict[str, Any]:
        """
        Returns settings as a dict with all secrets redacted.
        Safe to log, display in admin panels, or return via API.
        """
        data = self.dict()
        secret_fields = {
            "app_secret", "postgres_password", "redis_password",
            "qdrant_api_key", "gemini_api_key", "openai_api_key",
            "anthropic_api_key", "jwt_secret", "google_client_secret",
            "admin_secret_key",
        }
        for field in secret_fields:
            if field in data and data[field] is not None:
                data[field] = "**REDACTED**"
        return data

    # -----------------------------------------------------------------------
    # PYDANTIC CONFIG
    # -----------------------------------------------------------------------
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False          # Accept UPPER_CASE env vars
        validate_assignment = True      # Re-validate on attribute assignment
        use_enum_values = True
        # Allow extra fields (future env vars won't break startup)
        extra = "ignore"


# ===========================================================================
# Singleton accessor
# ===========================================================================
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.

    Uses lru_cache so the .env file is read and validated exactly once
    at startup. Call get_settings() anywhere in the codebase — it's
    always the same validated instance.

    Raises:
        pydantic.ValidationError: If required env vars are missing or invalid.
        SystemExit: If a critical misconfiguration is detected in production.
    """
    try:
        settings = Settings()
        _validate_critical_settings(settings)
        _configure_logging(settings)
        _log.info(
            "Configuration loaded successfully",
            extra={
                "env": settings.app_env,
                "version": settings.app_version,
                "providers": settings.active_llm_providers,
                "features": {
                    "rag": settings.feature_rag,
                    "agent": settings.feature_agent,
                    "streaming": settings.feature_streaming,
                    "memory": settings.feature_long_term_memory,
                },
            },
        )
        return settings
    except Exception as exc:
        _log.critical("Failed to load configuration: %s", exc)
        # In production, a bad config should prevent startup entirely
        if os.getenv("APP_ENV", "development") == "production":
            sys.exit(1)
        raise


def _validate_critical_settings(settings: Settings) -> None:
    """
    Perform cross-field validations that Pydantic validators can't easily
    handle (e.g. checking combinations of optional fields).

    Raises:
        ValueError: On any critical misconfiguration.
    """
    errors: List[str] = []

    # At least one LLM provider must be configured
    if not settings.active_llm_providers:
        errors.append(
            "No LLM provider configured. Set at least one of: "
            "GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or enable OLLAMA."
        )

    # Google OAuth requires both client ID and secret
    if settings.feature_google_oauth:
        if not settings.google_client_id or not settings.google_client_secret:
            errors.append(
                "FEATURE_GOOGLE_OAUTH=true requires both "
                "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
            )

    # In production, secrets must not be default/weak values
    if settings.is_production:
        weak_secrets = {"secret", "password", "changeme", "example", "test", "dev"}
        app_secret_val = settings.app_secret.get_secret_value().lower()
        jwt_secret_val = settings.jwt_secret.get_secret_value().lower()

        if any(w in app_secret_val for w in weak_secrets):
            errors.append("APP_SECRET appears weak. Use a strong random value in production.")

        if any(w in jwt_secret_val for w in weak_secrets):
            errors.append("JWT_SECRET appears weak. Use a strong random value in production.")

        if len(settings.jwt_secret.get_secret_value()) < 32:
            errors.append("JWT_SECRET must be at least 32 characters in production.")

        if settings.debug:
            errors.append("DEBUG must be false in production.")

    if errors:
        error_msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"Configuration errors:\n{error_msg}")


def _configure_logging(settings: Settings) -> None:
    """
    Configure root logger based on settings.
    JSON structured logging in production, human-readable in development.
    """
    log_format = (
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        if settings.is_production
        else "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ===========================================================================
# Utility: generate a strong secret (use during first-time setup)
# ===========================================================================
def generate_secret(length: int = 64) -> str:
    """Generate a cryptographically strong random secret string."""
    return secrets.token_urlsafe(length)


# ===========================================================================
# Dev convenience: print safe config summary
# ===========================================================================
if __name__ == "__main__":
    import json
    cfg = get_settings()
    print(json.dumps(cfg.as_safe_dict(), indent=2, default=str))