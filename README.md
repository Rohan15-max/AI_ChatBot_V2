# DevMentor AI 🧠

<div align="center">

**A production-ready, self-hostable General Purpose AI Platform**

[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-production-blue)](docker-compose.prod.yml)
[![Flask](https://img.shields.io/badge/flask-3.0-lightgrey)](https://flask.palletsprojects.com/)
[![Qdrant](https://img.shields.io/badge/qdrant-vector--db-purple)](https://qdrant.tech/)
[![Redis](https://img.shields.io/badge/redis-cache-red)](https://redis.io/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-326CE5)](https://kubernetes.io/)
[![Prometheus](https://img.shields.io/badge/prometheus-monitoring-E6522C)](https://prometheus.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#-contributing)

[Overview](#-what-is-devmentor-ai) • [Features](#-features) • [Roadmap](#%EF%B8%8F-roadmap) • [Architecture](#%EF%B8%8F-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Security](#-security) • [Deployment](#-deployment)

</div>

---

## 🌟 What is DevMentor AI?

DevMentor AI is a self-hostable **General Purpose AI Platform** — an enterprise-grade, data-isolated AI orchestration platform designed for secure, self-hosted deployment. It serves as a unified gateway that bridges proprietary large language models (LLMs) with private infrastructure, enabling organizations to deploy advanced AI workflows without exposing sensitive data to third-party ecosystems, running entirely on infrastructure you control. No third-party data pipes, no vendor lock-in: your own Postgres, Redis, and Qdrant, fronted by a Flask application that intelligently routes requests across Gemini, OpenAI, and Anthropic based on cost, latency, and availability.

It's built on the five capabilities every general-purpose assistant needs at its core:

- **Cost-aware multi-model routing** — automatic fallback across providers to maximize uptime and minimize spend
- **Hybrid Retrieval-Augmented Generation (RAG)** — grounded answers from a Qdrant vector core fused with BM25 keyword search
- **Long-term, decaying memory** — persistent, importance-weighted recall of facts about each user across sessions
- **Autonomous agent loop** — sandboxed, ReAct-style multi-step tool execution
- **High-throughput streaming** — token-by-token delivery over WebSocket, tuned for minimal time-to-first-token

Everything else on the roadmap — vision, voice, code execution, integrations — grows outward from that foundation. See [Roadmap](#%EF%B8%8F-roadmap) for what's shipped versus what's next.

> Designed and iterated on by a 19-year-old solo developer from India, hardened through real production debugging passes rather than written once and left alone.
> *"Don't wait until it's perfect. Ship it, improve it, repeat."*

---

## 🎯 Why Self-Host DevMentor AI?

| | Hosted AI SaaS | DevMentor AI (self-hosted) |
|---|---|---|
| **Data residency** | Leaves your infrastructure | Stays in your Postgres / Qdrant |
| **Provider lock-in** | Single vendor | Multi-provider fallback (Gemini / OpenAI / Anthropic) |
| **Cost control** | Fixed per-seat or usage pricing | You pay providers directly; semantic caching cuts redundant calls |
| **Customization** | Limited to vendor's feature set | Full source access — extend agents, tools, and RAG pipeline freely |
| **Compliance** | Depends on vendor's certifications | You control the audit trail, retention, and deletion policies |

---

## 🚀 Features

### 🧠 AI & Intelligence
- **Multi-model LLM routing** — Gemini, OpenAI, and Anthropic with cost-aware selection and automatic fallback when a provider fails or is rate-limited
- **Hybrid RAG pipeline** — Qdrant vector search + BM25 keyword search, fused via Reciprocal Rank Fusion, with cross-encoder reranking
- **Long-term memory** — recalls relevant facts about a user across sessions, with importance scoring and time-based decay
- **Context compression** — long conversations are automatically summarized past a message-count threshold, keeping prompt size and cost bounded without losing recent context
- **Semantic response caching** — skips redundant LLM calls for repeated or near-duplicate questions
- **Autonomous agent mode** — multi-step ReAct-style task execution with tool calling (web search, calculator, Wikipedia, weather, unit conversion, JSON formatting — growing)
- **Real-time streaming** — token-by-token delivery over WebSocket, with mid-generation stop support

### 🔐 Security
- JWT authentication (access + refresh tokens) with Redis-backed instant revocation, plus Google OAuth and API key support
- Prompt injection detection with severity scoring, applied to every chat, RAG, and agent request
- PII redaction across logs and stored data
- Per-user and per-IP rate limiting with sliding-window enforcement
- Full audit logging for every sensitive action (auth, deletions, admin actions)
- Account lockout after repeated failed login attempts

### 📊 Observability
- Prometheus metrics with Grafana dashboards
- Structured logging with automatic PII redaction
- Sentry error tracking integration
- Liveness, readiness, and detailed dependency health checks (database, Redis, Qdrant, Celery, JWT blacklist)

### ☁️ Infrastructure
- Docker Compose for both local development and production
- Kubernetes manifests with HPA autoscaling and an Nginx ingress
- Terraform for cloud provisioning
- Nginx reverse proxy with WebSocket-aware routing and SSL termination
- Automated PostgreSQL, Qdrant, and Redis backup and restore scripts
- Celery + Redis for background tasks (RAG indexing, document processing, scheduled maintenance)

---

## 🗺️ Roadmap

DevMentor AI's foundation — the part every general-purpose assistant is built on top of — is live today: multi-model chat with automatic fallback, hybrid RAG, long-term memory, an autonomous agent, and real-time streaming, all behind enterprise-grade auth and security.

The platform is actively expanding outward from that foundation toward full general-purpose parity with assistants like Claude, ChatGPT, and Gemini:

| Capability area | Status |
|---|---|
| Multi-model chat with fallback | ✅ Shipped |
| Hybrid RAG over your documents | ✅ Shipped |
| Long-term memory | ✅ Shipped |
| Context compression for long conversations | ✅ Shipped |
| Semantic response caching | ✅ Shipped |
| Autonomous agent + tool calling | ✅ Shipped (calculator, web search — growing) |
| Real-time streaming | ✅ Shipped |
| Code execution sandbox | 🔜 Planned |
| Image generation & vision | 🔜 Planned |
| Voice input / output | 🔜 Planned |
| Spreadsheet & document analysis | 🔜 Planned |
| Third-party integrations (Slack, Notion, etc.) | 🔭 Future |
| Multi-tenant billing & subscription tiers | 🔭 Future |
| Local/open-weight model support (Ollama) | 🔭 Future |

This table is updated as features ship. If something above is marked planned and you'd like to contribute, see [Contributing](#-contributing).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│              Browser (HTTP / WebSocket / OAuth)               │
└───────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      GATEWAY LAYER                            │
│              Nginx (SSL Termination + Reverse Proxy)           │
└───────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                          │
│          Flask + Flask-SocketIO (gthread workers)              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ Auth Module │  │ RAG Pipeline │  │   Agent Loop      │    │
│  │ JWT / OAuth │  │ Hybrid Search│  │   Tool Calling     │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ Cost Router │  │ Semantic     │  │  Security Suite   │    │
│  │ Multi-model │  │ Cache        │  │  PII + Injection   │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
└───────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      WORKER LAYER                              │
│              Celery (Async Tasks + RAG Indexing)                │
└──────────┬──────────────────┬────────────────────────────────┘
           │                  │
           ▼                  ▼
┌──────────────────┐  ┌──────────────────────────────────────┐
│  Redis            │  │         DATA LAYER                    │
│  - Task Queue      │  │  ┌────────────┐  ┌────────────────┐ │
│  - Rate Limiter    │  │  │ PostgreSQL │  │    Qdrant       │ │
│  - JWT Blacklist   │  │  │ (State +   │  │ (Vector Store)  │ │
│  - Semantic Cache   │  │  │  History)  │  │                │ │
└──────────────────┘  │  └────────────┘  └────────────────┘  │
                      └──────────────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────────┐
                      │           LLM PROVIDERS                │
                      │      Gemini │ OpenAI │ Anthropic        │
                      └──────────────────────────────────────┘
```

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| API framework | Flask 3.0 + Flask-SocketIO (gthread workers) |
| Relational database | PostgreSQL + SQLAlchemy ORM + Alembic migrations |
| Vector database | Qdrant |
| Cache / queues | Redis (rate limiting, JWT blacklist, semantic cache, Celery broker) |
| Background jobs | Celery |
| LLM providers | Gemini, OpenAI, Anthropic (cost-aware routing + fallback chain) |
| Auth | JWT (access + refresh), Google OAuth, API keys |
| Reverse proxy | Nginx (SSL termination, WebSocket-aware routing) |
| Monitoring | Prometheus + Grafana + Sentry |
| Orchestration | Docker Compose (dev/prod), Kubernetes (HPA + ingress), Terraform |
| Frontend | Server-rendered SPA (`templates/index.html`) with a custom design system, Socket.IO client for streaming |

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker | 20.10+ | Required for containerized deployment |
| Docker Compose | 2.0+ | Required |
| Python | 3.11 | Local (non-Docker) development only |
| RAM | 8 GB minimum | 16 GB recommended |
| CPU | 4 cores recommended | — |
| LLM API key | Gemini (required) | OpenAI / Anthropic optional, enable provider fallback when set |

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key |
| `POSTGRES_USER` | ✅ Yes | PostgreSQL username |
| `POSTGRES_PASSWORD` | ✅ Yes | PostgreSQL password |
| `POSTGRES_DB` | ✅ Yes | PostgreSQL database name |
| `REDIS_URL` | ✅ Yes | Redis connection string |
| `QDRANT_HOST` | ✅ Yes | Qdrant host |
| `QDRANT_PORT` | ✅ Yes | Qdrant port (default: `6333`) |
| `JWT_SECRET` | ✅ Yes | Random secret, 32+ characters |
| `APP_SECRET` | ✅ Yes | Flask session secret key |
| `OPENAI_API_KEY` | ⚪ Optional | Enables OpenAI in the fallback chain |
| `ANTHROPIC_API_KEY` | ⚪ Optional | Enables Anthropic in the fallback chain |
| `SERPER_API_KEY` | ⚪ Optional | Real web search results for agent mode; falls back to a limited instant-answer API if unset |
| `GOOGLE_CLIENT_ID` | ⚪ Optional | Google OAuth sign-in |
| `SENTRY_DSN` | ⚪ Optional | Error tracking |

See `.env.example` for the complete list.

> **Note on production deployment:** `config.py` reads individual `POSTGRES_HOST` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` variables — a combined `DATABASE_URL` is not read. Set the individual vars.

---

## 🐳 Quick Start

### Option 1 — Docker (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/devmentor-ai.git
cd devmentor-ai

# 2. Configure environment
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY and secrets

# 3. Start all services (development configuration)
docker compose up --build

# 4. Check all services are healthy
docker compose ps
```

**Access points (development, via `docker-compose.yml`):**

| Service | URL |
|---|---|
| Web UI | http://localhost:8000 |

**Access points (production, via `docker-compose.prod.yml`):**

| Service | URL |
|---|---|
| Web UI | http://localhost:5000 (bound to `127.0.0.1` only — put Nginx in front for external access) |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

```bash
# Stop everything
docker compose down
```

### Option 2 — Local development (no Docker for the app itself)

Postgres, Redis, and Qdrant still need to run somewhere reachable — the steps below run them in Docker and the Flask app directly on your machine.

```bash
# 1. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Start infrastructure services only
docker compose up -d postgres redis qdrant

# 4. Configure environment
cp .env.example .env
# Set POSTGRES_HOST, REDIS_URL host, and QDRANT_HOST to localhost

# 5. Initialize and run database migrations (first time only)
alembic init alembic
# Then add the project-settings hook described at the top of alembic.ini
alembic upgrade head

# 6. Start the application
python app.py
```

---

## 📡 API Reference

> Every endpoint below is prefixed with `/api/v1`. All routes except `/api/v1/health` require authentication via `Authorization: Bearer <jwt>`.

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chat` | Standard chat, routed through the multi-provider fallback chain |
| `POST` | `/api/v1/chat/stream` | Starts a streamed response delivered over the existing WebSocket connection |
| `POST` | `/api/v1/chat/stream/<stream_id>/stop` | Requests cancellation of an in-progress stream |
| `POST` | `/api/v1/chat-rag` | RAG-augmented chat, grounded in the user's uploaded documents |
| `POST` | `/api/v1/agent` | Runs an autonomous agent task (general, research, or coding mode) |
| `WS` | `/socket.io/` | WebSocket endpoint for streamed tokens |

### Knowledge base

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/upload_docs` | Upload documents to the knowledge base |
| `GET` | `/api/v1/rag/status` | Get the current RAG index status |
| `POST` | `/api/v1/rag/rebuild` | Manually trigger a full index rebuild |
| `DELETE` | `/api/v1/rag/clear` | Clear the user's knowledge base |

### Conversations

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/threads` | List the user's conversation threads |
| `GET` | `/api/v1/threads/<id>` | Get a specific thread and its messages |
| `PATCH` | `/api/v1/threads/<id>/rename` | Rename a thread |
| `DELETE` | `/api/v1/threads/<id>` | Delete a thread |
| `GET` | `/api/v1/threads/<id>/export` | Export a thread as Markdown |
| `GET` | `/api/v1/search` | Full-text search across the user's messages |

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Register a new account |
| `POST` | `/api/v1/auth/login` | Log in with username and password |
| `POST` | `/api/v1/auth/google` | Google OAuth login |
| `POST` | `/api/v1/auth/refresh` | Exchange a refresh token for a new access token |
| `POST` | `/api/v1/auth/logout` | Log out and revoke the current token |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check, no authentication required |
| `GET` | `/health/readiness` | Readiness check — verifies database connectivity |
| `GET` | `/health/detailed` | Full dependency health (database, Redis, Qdrant, Celery) |
| `GET` | `/metrics` | Prometheus metrics |

---

## 🔒 Security

```
Request → Rate Limiter → Auth (JWT / Session) → Prompt Injection Check
        → Input Sanitization → LLM → PII Redaction (logs) → Audit Logger → Client
```

| Feature | Implementation |
|---|---|
| Authentication | JWT (access + refresh) + session + Google OAuth |
| Token revocation | Redis-backed JWT blacklist, checked on every request |
| Prompt injection detection | `security/prompt_injection.py` — pattern + heuristic scoring, applied on chat, RAG, and agent endpoints |
| PII protection | `security/pii_redactor.py` — redacts before logging |
| Audit trail | `security/audit_logger.py` — every sensitive action logged to file and database |
| Rate limiting | Per-user and per-IP, sliding window (40 req/min on chat by default — see `rate_limiter.py`) |
| Transport security | HTTPS via Nginx + your certificate provider of choice |

### Security Checklist Before Going to Production

- [ ] Rotate `JWT_SECRET` and `APP_SECRET` away from any development defaults
- [ ] Set `POSTGRES_PASSWORD` and `REDIS_URL` credentials to strong, unique values
- [ ] Put Nginx (or another TLS-terminating proxy) in front of the app — the production Compose file binds the app to `127.0.0.1` only, by design
- [ ] Confirm `SENTRY_DSN` is set so production errors aren't silently lost
- [ ] Review `security/audit_logger.py` retention settings against your compliance requirements
- [ ] Schedule `scripts/backup.sh` via cron and periodically test `scripts/restore.sh`

---

## 📈 Monitoring

| Tool | URL | Purpose |
|---|---|---|
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Dashboards |
| Sentry | Via DSN | Error tracking |

**Key metrics tracked:**
- `devmentor_chat_requests_total` — total chat requests, labeled by model/provider/status
- `devmentor_chat_duration_seconds` — response latency
- `devmentor_rag_cache_hits_total` — semantic cache efficiency
- `devmentor_token_usage_total` — LLM token consumption
- `devmentor_rate_limit_exceeded_total` — rate limit violations
- `devmentor_active_users` — active users in the last 5 minutes

Provider-down alerts for both Gemini and Anthropic are pre-configured in `ops/monitoring/alert_rules.yml`.

---

## 🗄️ Backup & Restore

```bash
# Full backup (PostgreSQL + Qdrant + Redis)
./scripts/backup.sh

# Restore from the most recent backup
./scripts/restore.sh

# Rotate the JWT signing secret
python scripts/rotate_secrets.py
```

> Rotating the JWT secret edits `.env` on disk but does not hot-reload a running process — restart the app (and any Celery workers) afterward for the new secret to take effect.

Schedule backups with cron:
```bash
0 2 * * * /path/to/scripts/backup.sh
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific suite
pytest tests/test_chat.py -v
pytest tests/test_security.py -v
pytest tests/test_rate_limit.py -v
pytest tests/test_long_memory.py -v
pytest tests/test_fallback.py -v

# Load testing
locust -f tests/load_test/locustfile.py --host http://localhost:8000
```

> `test_rate_limit.py` and `test_long_memory.py` require a reachable Redis / Qdrant respectively and will skip themselves with a clear reason if those aren't available, rather than passing for the wrong reason.

---

## 🚢 Deployment

### Docker Compose (single server)
```bash
docker compose -f docker-compose.prod.yml up -d
```

### Kubernetes
```bash
# Deploy all manifests
kubectl apply -f ops/k8s/

# Check deployment status
kubectl get pods
kubectl get services

# Scale manually (autoscaling is also configured via ops/k8s/hpa.yaml)
kubectl scale deployment ai-chatbot --replicas=3
```

### Terraform (cloud provisioning)
```bash
cd ops/terraform
terraform init
terraform plan
terraform apply
```

---

## 🐛 Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| WebSocket not connecting / intermittent drops | Wrong Gunicorn worker class for the configured Socket.IO async mode | Use `--worker-class gthread` — `websocket_handler.py` initializes Flask-SocketIO with `async_mode="threading"`, which conflicts with eventlet/gevent's monkey-patching |
| App can't reach the database in production | `DATABASE_URL`-style env vars aren't read by `config.py` | Set the individual `POSTGRES_HOST` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` vars instead |
| RAG returns nothing | No documents uploaded yet | Upload via `/api/v1/upload_docs`, then check `/api/v1/rag/status` |
| 429 Too Many Requests | Rate limit exceeded | Wait for the window to reset, or adjust the relevant `rate_limit_*` settings |
| Redis connection refused | Redis isn't running | `docker compose up -d redis` |
| JWT expired | Token lifetime exceeded | Call `/api/v1/auth/refresh`, or log in again |
| Agent web search returns weak results | No `SERPER_API_KEY` configured | Add a free Serper.dev key to `.env` — search falls back to a limited instant-answer API without one |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch — `git checkout -b feature/your-feature`
3. Write tests for your changes
4. Run the test suite — `pytest tests/ -v`
5. Commit your changes — `git commit -m "Add your feature"`
6. Push your branch — `git push origin feature/your-feature`
7. Open a pull request against `develop`

Interested in pushing the platform toward full general-purpose parity? Check the [Roadmap](#%EF%B8%8F-roadmap) for what's planned next — code execution, vision, voice, and broader integrations are all open territory.

---

## 📁 Project Structure

```
devmentor-ai/
├── ai/                      # AI/ML modules
│   ├── cost_router.py       # Cost-aware model selection
│   ├── reranker.py          # Cross-encoder reranking
│   └── semantic_cache.py    # Semantic similarity caching
├── security/                # Security suite
│   ├── audit_logger.py      # Action audit logging
│   ├── jwt_blacklist.py     # Token revocation
│   ├── pii_redactor.py      # PII redaction
│   └── prompt_injection.py  # Injection detection
├── ops/                     # Infrastructure
│   ├── k8s/                 # Kubernetes manifests
│   ├── monitoring/          # Prometheus alert rules
│   ├── logging/             # Fluent Bit config
│   └── terraform/           # IaC
├── scripts/                 # Utility scripts
│   ├── backup.sh
│   ├── restore.sh
│   ├── migrate.sh
│   ├── seed_test_data.py
│   └── rotate_secrets.py
├── tests/                   # Test suite
│   ├── load_test/           # Locust load tests
│   └── *.py                 # Unit and integration tests
├── templates/                # Frontend
│   └── index.html            # Web UI
├── static/                   # Frontend assets
│   └── design-system.css
├── app.py                    # Application entry point
├── config.py                 # Configuration
├── context_compressor.py     # Long-conversation summarization
├── agent_loop.py              # Agent execution loop
├── agent_tools.py             # Agent tool definitions
├── websocket_handler.py       # WebSocket streaming
├── long_term_memory.py        # Cross-session memory
├── model_router.py            # Multi-provider LLM routing
├── qdrant_wrapper.py           # Qdrant vector DB wrapper
├── database.py                 # Database models and connections
├── analytics.py                # Usage analytics
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
└── requirements.txt
```

---

## 📄 License

Proprietary — All rights reserved.
Contact the maintainer for licensing inquiries.

---

## 📬 Contact

**Rohan Gaud**
📧 rohangaud2007@gmail.com
🐙 [GitHub](https://github.com/your-org/devmentor-ai)

---

<div align="center">

Built with ❤️ passion and consistency by a 19-year-old developer from India.

*"Don't wait until it's perfect. Ship it, improve it, repeat."*

⭐ Star this repo if you found it useful!

</div>