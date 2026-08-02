#!/usr/bin/env python3
import os
import stat

folders = [
    "scripts", "tests", "tests/load_test", "security", "ai",
    "ops/terraform", "ops/k8s", "ops/monitoring", "ops/logging",
    ".github/workflows"
]

files_content = {}

# ----- ROOT -----
files_content[".env.example"] = '''# Environment variables template
APP_ENV=production
APP_SECRET=change_this
LOG_LEVEL=INFO
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=aichat
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=change_me
REDIS_URL=redis://redis:6379/0
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
RATE_LIMIT_PER_USER=100
RATE_LIMIT_WINDOW=60
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
JWT_SECRET=change_this
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
PROMETHEUS_PORT=8000
'''

files_content["config.py"] = '''from functools import lru_cache
from pydantic import BaseSettings, RedisDsn

class Settings(BaseSettings):
    app_env: str = "development"
    app_secret: str
    log_level: str = "INFO"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "aichat"
    postgres_user: str
    postgres_password: str
    redis_url: RedisDsn = "redis://localhost:6379/0"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    openai_api_key: str
    anthropic_api_key: str | None = None
    rate_limit_per_user: int = 100
    rate_limit_window: int = 60
    celery_broker_url: RedisDsn = "redis://localhost:6379/1"
    celery_result_backend: RedisDsn = "redis://localhost:6379/2"
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    prometheus_port: int = 8000

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
'''

files_content["docker-compose.prod.yml"] = '''version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    environment:
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
      - qdrant
    command: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
'''

files_content["prometheus.yml"] = '''global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'ai_platform'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
'''

files_content["pyproject.toml"] = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-chatbot-v2"
version = "2.0.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.23.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
    "qdrant-client>=1.7.0",
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "psycopg2-binary>=2.9.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "pydantic[dotenv]>=2.0.0",
    "python-jose[cryptography]>=3.3.0",
    "prometheus-client>=0.18.0",
    "python-json-logger>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black>=23.0", "mypy>=1.0", "bandit>=1.7", "pre-commit>=3.0"]

[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
'''

files_content["alembic.ini"] = '''[alembic]
script_location = alembic
sqlalchemy.url = postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}/${POSTGRES_DB}
'''

files_content["README.md"] = '''# AI Chatbot Platform v2 - Production Edition
See docs for setup. Use docker-compose.prod.yml to start.
'''

files_content["requirements-dev.txt"] = '''pytest>=7.0
black>=23.0
mypy>=1.0
bandit>=1.7
'''

files_content["Makefile"] = '''dev:
	pip install -r requirements.txt && python app.py
test:
	pytest tests/
'''

files_content[".gitignore"] = '''.env
*.db
__pycache__/
.venv/
rag_storage/
logs/
'''

# ----- .github/workflows/ci.yml (shortened) -----
files_content[".github/workflows/ci.yml"] = '''name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt pytest
      - name: Test
        run: pytest tests/
'''

# ----- scripts/ -----
files_content["scripts/backup.sh"] = '''#!/bin/bash
echo "Backup script placeholder"
'''

files_content["scripts/restore.sh"] = '''#!/bin/bash
echo "Restore script placeholder"
'''

files_content["scripts/migrate.sh"] = '''#!/bin/bash
alembic upgrade head
'''

files_content["scripts/seed_test_data.py"] = '''print("Seeding test data")'''
files_content["scripts/rotate_secrets.py"] = '''print("Rotating secrets")'''

# ----- tests/ -----
files_content["tests/test_chat.py"] = '''from fastapi.testclient import TestClient
from app import app
client = TestClient(app)
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
'''

files_content["tests/test_rate_limit.py"] = '''# Rate limit test placeholder'''
files_content["tests/test_security.py"] = '''# Security test placeholder'''
files_content["tests/test_fallback.py"] = '''# Fallback test placeholder'''
files_content["tests/test_long_memory.py"] = '''# Memory test placeholder'''
files_content["tests/load_test/locustfile.py"] = '''from locust import HttpUser, task
class User(HttpUser):
    @task
    def chat(self):
        self.client.post("/chat", json={"message":"hi"})
'''

# ----- security/ -----
files_content["security/pii_redactor.py"] = "def redact_pii(t): return t"
files_content["security/audit_logger.py"] = "def log_audit(*a,**k): pass"
files_content["security/jwt_blacklist.py"] = "def revoke_token(*a): pass"
files_content["security/prompt_injection.py"] = "def detect_injection(t): return False"

# ----- ai/ -----
files_content["ai/semantic_cache.py"] = "def semantic_cache_key(q): return None"
files_content["ai/reranker.py"] = "def rerank_documents(q,d): return d"
files_content["ai/cost_router.py"] = "class CostAwareRouter: def route(self,u,q,t): return 'gpt-3.5'"

# ----- ops/ -----
files_content["ops/terraform/main.tf"] = '# Terraform config placeholder'
files_content["ops/k8s/deployment.yaml"] = '# K8s deployment'
files_content["ops/k8s/ingress.yaml"] = '# K8s ingress'
files_content["ops/k8s/hpa.yaml"] = '# K8s hpa'
files_content["ops/monitoring/alert_rules.yml"] = '# Prometheus alerts'
files_content["ops/logging/fluentbit.conf"] = '# FluentBit config'

# ----------------------------------------------------------------------
def main():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 {folder}")
    for path, content in files_content.items():
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if path.endswith(".sh"):
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        print(f"📄 {path}")
    print("\n✅ All files created.\n⚠️ Manually replace app.py and templates/index.html with the upgraded versions.")

if __name__ == "__main__":
    main()