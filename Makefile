# ============================================================
# Makefile — shortcut commands for your Docker AI app
# Usage: make <command>
# ============================================================

.PHONY: build up down logs shell db-shell restart clean rebuild

# Build all images
build:
	docker compose build --no-cache

# Start everything
up:
	docker compose up -d

# Start and show logs
up-logs:
	docker compose up

# Stop everything
down:
	docker compose down

# Stop and remove volumes (DESTRUCTIVE — wipes DB data)
down-clean:
	docker compose down -v

# View logs
logs:
	docker compose logs -f app

logs-all:
	docker compose logs -f

# Shell into the app container
shell:
	docker compose exec app bash

# Shell into the DB
# FIX: was `exec db`, but no service named "db" exists in either
# docker-compose.yml or docker-compose.prod.yml — the Postgres service is
# named "postgres" in both files. This previously failed with something
# like "service db is not running" every time.
db-shell:
	docker compose exec postgres psql -U $${POSTGRES_USER} -d $${POSTGRES_DB}

# Restart just the app (after code changes)
restart:
	docker compose restart app celery_worker

# Run migrations manually
migrate:
	docker compose exec app alembic upgrade head

# Full rebuild (when requirements.txt changes)
rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d

# Check health of all containers
# NOTE: this hits port 5000 inside the container, which matches the
# Dockerfile's actual gunicorn bind — left as-is since the previous bug was
# in docker-compose.yml's HOST-side port mapping, not in the container's
# own internal port, which `docker compose exec` talks to directly.
health:
	docker compose ps
	docker compose exec app curl -s http://localhost:5000/health

# Clean unused Docker resources
clean:
	docker system prune -f
	docker volume prune -f