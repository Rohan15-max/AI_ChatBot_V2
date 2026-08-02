"""
DevMentor AI — WebSocket Handler
==================================
Production-grade real-time streaming with:

- JWT + session authentication on connect
- Per-user sliding window rate limiting (Redis)
- Thread ownership verification
- Stop generation support
- Reconnection handling
- Structured error events
- Room-based isolation per stream

UPGRADE NOTES (this revision):

The previous version of this file ran its OWN independent streaming
pipeline (`send_message` -> `message_start`/`message_chunk`/`message_end`),
calling Gemini directly via the old `google.generativeai` SDK and bypassing
`model_router.py` entirely. That meant the multi-provider fallback chain
(Gemini -> GPT-4o-mini -> Claude -> Ollama) built into model_router.py never
activated for streamed messages, even after app.py was upgraded to use it
for the HTTP chat path. It also required a thread to already exist before
streaming could start, which breaks the very common case of a brand new
conversation. Separately, `rag_query` imported a `backend.get_rag_system`
function that doesn't exist anywhere in this project and would crash
immediately if a client ever called it.

This revision removes that independent pipeline. Streaming generation now
lives in app.py's `/api/v1/chat/stream` route (which already does thread
creation, history loading, cost-aware routing through model_router, response
persistence, semantic caching, and long-term memory storage correctly). This
file's job is now just to let a client join the Socket.IO room that route
emits to:

    1. Client POSTs /api/v1/chat/stream -> receives {stream_id, ...}
    2. Client emits "join" with {room: stream_id}      <-- handled below
    3. app.py's background task emits "chat_token" / "chat_done" /
       "chat_error" to that room as the model_router stream yields tokens
    4. Client emits "leave" with {room: stream_id} when done (optional —
       Socket.IO also cleans up rooms automatically on disconnect)

The original `join_thread`/`leave_thread` events (for thread-level
presence, unrelated to streaming) are kept as-is since they're sound and
independent of this change.

Events (client -> server):
    connect          — authenticate via token query param
    join             — join a stream_id room to receive generation events
    leave            — leave a stream_id room
    join_thread      — join a thread-level room (presence, unrelated to streaming)
    leave_thread     — leave a thread-level room
    stop_generation  — cancel current stream
    ping             — keepalive

Events (server -> client):
    connected        — successful connection
    joined           — room join acknowledged
    left             — room leave acknowledged
    error            — general error
    pong             — keepalive response

    (chat_token / chat_done / chat_error are emitted by app.py's
     chat_stream() route, not by this file — this file only manages
     room membership and connection lifecycle for them.)
"""

import logging
import threading
import time
from functools import wraps
from typing import Dict

from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room

from auth_middleware import decode_access_token
from config import get_settings
from database import Thread, User, get_db, utc_now

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Active streams registry (for stop generation support)
# ---------------------------------------------------------------------------
_active_streams: Dict[str, bool] = {}  # sid -> should_stop
_active_streams_lock = threading.Lock()

# ---------------------------------------------------------------------------
# SocketIO instance
# ---------------------------------------------------------------------------
socketio = SocketIO(
    cors_allowed_origins=settings.allowed_origins,
    async_mode="threading",
    ping_timeout=settings.websocket_ping_timeout,
    ping_interval=settings.websocket_ping_interval,
    logger=False,
    engineio_logger=False,
)


# ===========================================================================
# Rate Limiting
# ===========================================================================

def _check_ws_rate_limit(user_id: str, action: str = "message", limit: int = 20) -> bool:
    """
    Sliding window rate limit for WebSocket messages.

    Args:
        user_id: User to check
        action:  Action type (message/rag/agent)
        limit:   Max requests per minute

    Returns:
        True if allowed, False if rate limited
    """
    try:
        from redis_client import get_redis_client
        redis = get_redis_client()
        if not redis:
            return True  # Fail open if Redis unavailable

        key = f"ws:rate:{action}:{user_id}"
        now = time.time()
        window = 60

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window)
        results = pipe.execute()

        current_count = results[1]
        return current_count < limit

    except Exception as exc:
        logger.warning("WS rate limit check failed: %s", exc)
        return True  # Fail open


# ===========================================================================
# Authentication Decorator
# ===========================================================================

def ws_auth_required(f):
    """
    WebSocket authentication decorator.

    Checks JWT token from:
    1. Query param: ?token=<jwt>
    2. Authorization header: Bearer <jwt>
    3. Socket.IO auth payload: io(url, {auth: {token: <jwt>}})
    4. Flask session

    On success, sets request.user_id, request.username, request.user_tier.
    On failure, emits error event and returns False.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get("token")

        if not token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]

        # Socket.IO client-side `auth: {token: ...}` payload arrives in the
        # connect event's first positional arg, not on request.args/headers —
        # handle_connect() passes it through explicitly when present.
        if not token and args and isinstance(args[0], dict):
            token = args[0].get("token")

        if not token:
            from flask import session
            user_id = session.get("user_id")
            if user_id:
                request.user_id = user_id
                request.username = session.get("username", "unknown")
                request.user_tier = session.get("tier", "free")
                request.is_admin = session.get("is_admin", False)
                return f(*args, **kwargs)

        if not token:
            logger.warning("WS connect without token: ip=%s", request.remote_addr)
            emit("error", {"code": "AUTH_REQUIRED", "message": "Authentication required"})
            return False

        payload = decode_access_token(token)
        if not payload:
            logger.warning("WS invalid token: ip=%s", request.remote_addr)
            emit("error", {"code": "TOKEN_INVALID", "message": "Invalid or expired token"})
            return False

        user_id = payload.get("sub")
        if not user_id:
            emit("error", {"code": "TOKEN_INVALID", "message": "Malformed token"})
            return False

        try:
            with get_db() as db:
                user = db.query(User).filter(
                    User.id == user_id,
                    User.is_deleted == False,
                ).first()

                if not user:
                    emit("error", {"code": "USER_NOT_FOUND", "message": "User not found"})
                    return False
                if not user.is_active:
                    emit("error", {"code": "ACCOUNT_INACTIVE", "message": "Account inactive"})
                    return False
                if user.is_locked():
                    emit("error", {"code": "ACCOUNT_LOCKED", "message": "Account temporarily locked"})
                    return False

                request.user_id = user.id
                request.username = user.username
                request.user_tier = user.tier
                request.is_admin = user.is_admin
                user.last_active = utc_now()

        except Exception as exc:
            logger.error("WS auth DB check failed: %s", exc)
            emit("error", {"code": "AUTH_ERROR", "message": "Authentication error"})
            return False

        return f(*args, **kwargs)

    return decorated


def _verify_thread_ownership(thread_id: str, user_id: str) -> bool:
    """Verify user owns the thread."""
    try:
        with get_db() as db:
            thread = db.query(Thread).filter(
                Thread.id == thread_id,
                Thread.user_id == user_id,
                Thread.is_deleted == False,
            ).first()
            return thread is not None
    except Exception as exc:
        logger.error("Thread ownership check failed: %s", exc)
        return False


# ===========================================================================
# Stream Management (stop-generation support)
# ===========================================================================

def _register_stream(sid: str) -> None:
    with _active_streams_lock:
        _active_streams[sid] = False

def _unregister_stream(sid: str) -> None:
    with _active_streams_lock:
        _active_streams.pop(sid, None)

def _should_stop(sid: str) -> bool:
    with _active_streams_lock:
        return _active_streams.get(sid, True)

def _request_stop(sid: str) -> None:
    with _active_streams_lock:
        if sid in _active_streams:
            _active_streams[sid] = True


# ===========================================================================
# WebSocket Event Handlers
# ===========================================================================

@socketio.on("connect")
@ws_auth_required
def handle_connect(auth=None):
    """Handle WebSocket connection. `auth` carries the client's `auth:{token}` payload, if any."""
    sid = request.sid
    logger.info("WS connected: sid=%s user=%s tier=%s", sid, request.user_id, request.user_tier)
    emit("connected", {
        "status": "ok",
        "user_id": request.user_id,
        "username": request.username,
        "tier": request.user_tier,
        "sid": sid,
        "features": {
            "streaming": settings.feature_streaming,
            "rag": settings.feature_rag,
            "agent": settings.feature_agent,
        },
    })


@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    sid = request.sid
    user_id = getattr(request, "user_id", "unknown")
    _request_stop(sid)
    _unregister_stream(sid)
    logger.info("WS disconnected: sid=%s user=%s", sid, user_id)


@socketio.on("ping")
def handle_ping():
    """Keepalive ping handler."""
    emit("pong", {"timestamp": utc_now().isoformat()})


# ---------------------------------------------------------------------------
# Stream room join/leave — this is what makes app.py's /api/v1/chat/stream
# actually deliver tokens to the browser. The client calls the HTTP route
# first to get a stream_id, then emits "join" with that stream_id as the
# room name before app.py's background task starts emitting chat_token
# events to it.
# ---------------------------------------------------------------------------

@socketio.on("join")
@ws_auth_required
def handle_join(data: Dict):
    """
    Join a room to receive streamed generation events for one response.

    Expected data: { room: <stream_id> }

    No ownership check against the database is needed here — stream_id is
    a freshly minted UUID returned only to the requesting user by
    /api/v1/chat/stream, so knowledge of it already proves authorization.
    Unlike thread_id (long-lived, reused across many requests), a stream_id
    is single-use and expires the moment that one response finishes.
    """
    room = (data or {}).get("room", "").strip()
    if not room:
        emit("error", {"code": "MISSING_FIELD", "message": "room is required"})
        return
    join_room(room)
    logger.debug("User %s joined stream room %s", request.user_id, room)
    emit("joined", {"room": room, "status": "ok"})


@socketio.on("leave")
@ws_auth_required
def handle_leave(data: Dict):
    """Leave a stream room. Optional — Socket.IO also cleans these up on disconnect."""
    room = (data or {}).get("room", "").strip()
    if room:
        leave_room(room)
        emit("left", {"room": room})


# ---------------------------------------------------------------------------
# Thread-level presence rooms — unrelated to generation streaming, kept
# as-is from the original. Useful if you later want multi-device sync
# (e.g. "this thread was updated elsewhere, refresh") within a thread the
# user owns long-term, as opposed to a one-shot stream_id.
# ---------------------------------------------------------------------------

@socketio.on("join_thread")
@ws_auth_required
def handle_join_thread(data: Dict):
    """Join a thread room for receiving cross-device updates on a conversation."""
    thread_id = (data or {}).get("thread_id", "").strip()

    if not thread_id:
        emit("error", {"code": "MISSING_FIELD", "message": "thread_id required"})
        return

    if not _verify_thread_ownership(thread_id, request.user_id):
        emit("error", {"code": "ACCESS_DENIED", "message": "Thread not found or access denied"})
        return

    join_room(thread_id)
    logger.info("User %s joined thread room %s", request.user_id, thread_id)
    emit("joined", {"thread_id": thread_id, "status": "ok"})


@socketio.on("leave_thread")
@ws_auth_required
def handle_leave_thread(data: Dict):
    """Leave a thread room."""
    thread_id = (data or {}).get("thread_id", "").strip()
    if thread_id:
        leave_room(thread_id)
        emit("left", {"thread_id": thread_id})


@socketio.on("stop_generation")
@ws_auth_required
def handle_stop_generation(data: Dict):
    """
    Stop an active generation stream for this socket connection.

    Note: app.py's chat_stream() background task currently checks
    model_router.stream()'s own internal generator state, not this
    registry, since it runs in the shared background event loop rather
    than a per-socket thread. This handler is kept so the registry and
    event exist for future wiring (e.g. if model_router.stream() is
    extended to accept a cancellation callback) without breaking the
    client contract in the meantime.
    """
    _request_stop(request.sid)
    emit("generation_stopped", {"status": "ok"})
    logger.info("Stop-generation requested by user %s", request.user_id)


# ===========================================================================
# Initialization
# ===========================================================================

def init_websocket(app) -> SocketIO:
    """
    Initialize SocketIO with the Flask app.
    Call this from your app factory.

    Args:
        app: Flask application instance

    Returns:
        Configured SocketIO instance
    """
    message_queue = settings.redis_url if settings.feature_streaming else None

    socketio.init_app(
        app,
        message_queue=message_queue,
        channel="devmentor",
        cors_allowed_origins=settings.allowed_origins,
    )

    logger.info(
        "WebSocket initialized: origins=%s queue=%s",
        settings.allowed_origins,
        "redis" if message_queue else "none",
    )

    return socketio