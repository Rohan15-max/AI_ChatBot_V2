# ============================================================
# STAGE 1: Builder — installs all Python deps with build tools
# ============================================================
FROM python:3.11-slim AS builder

# System packages needed to COMPILE/install your requirements:
# - gcc, build-essential  → psycopg2-binary, bcrypt, pynacl
# - libpq-dev             → psycopg2-binary
# - libmagic1, libmagic-dev → python-magic (your magic>=0.4.27)
# - libssl-dev, libffi-dev → cryptography (used by PyJWT, passlib)
# - curl                  → healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    libmagic1 \
    libmagic-dev \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install dependencies (cached as a separate layer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ============================================================
# STAGE 2: Runtime — lean image, no build tools
# ============================================================
FROM python:3.11-slim AS runtime

# Only RUNTIME system libraries (no gcc, no -dev packages)
# libmagic1        → python-magic at runtime
# libpq5           → psycopg2 at runtime
# libgomp1         → llama-index / numpy thread support
# curl             → Docker HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY . .

# Create non-root user (security best practice)
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/logs /app/uploads /app/knowledge_base /app/rag_storage && \
    chown -R appuser:appuser /app
USER appuser

# Flask / gunicorn port
EXPOSE 5000

# Healthcheck — hits your /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use entrypoint script for migrations + startup
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# FIX: switched from --worker-class eventlet to gthread (threaded sync
# workers). websocket_handler.py explicitly initializes Flask-SocketIO with
# async_mode="threading" — eventlet and gevent both monkey-patch Python's
# threading/socket internals at import time, which conflicts with
# async_mode="threading" expecting *unpatched* native threads. app.py's
# shared background event loop (_BackgroundEventLoop) also depends on a
# real native thread running asyncio.run_forever() — eventlet's greenlet-
# based cooperative scheduling does not reliably coexist with that. gthread
# uses real OS threads, matching both of these correctly. --threads sets
# how many threads each worker process uses to serve concurrent requests.
CMD ["gunicorn", "--worker-class", "gthread", "-w", "1", "--threads", "8", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "app:app"]