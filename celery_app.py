"""
DevMentor AI — Celery Application Configuration
==================================================
Production-grade Celery setup with:

- Separate broker and result backend (correct Redis DB separation)
- Multi-queue routing for task prioritization
- Worker concurrency, prefetch, and time limit tuning
- Periodic task schedule (Celery Beat) for maintenance jobs
- Task acknowledgment configured for reliability (late ack)
- Connection retry on startup
- Result expiration to prevent Redis bloat
- Dead letter handling via max retries

Usage:
    # Start worker
    celery -A celery_app worker --loglevel=info -Q default,rag,email,maintenance,batch,webhooks

    # Start beat scheduler (periodic tasks)
    celery -A celery_app beat --loglevel=info
"""

import logging

from celery import Celery
from celery.schedules import crontab

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ===========================================================================
# Celery App Instance
# ===========================================================================

celery_app = Celery(
    settings.app_name.lower().replace(" ", "_"),
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "background_tasks",
    ],
)


# ===========================================================================
# Configuration
# ===========================================================================

celery_app.conf.update(
    # ---- Serialization ----
    task_serializer=settings.celery_task_serializer,
    accept_content=[settings.celery_task_serializer],
    result_serializer=settings.celery_result_serializer,

    # ---- Timezone ----
    timezone="UTC",
    enable_utc=True,

    # ---- Reliability ----
    task_track_started=True,
    task_acks_late=True,              # Ack after completion, not on receipt — survives worker crashes
    worker_prefetch_multiplier=1,     # Don't hoard tasks — fair distribution across workers
    task_reject_on_worker_lost=True,  # Re-queue task if worker dies mid-execution

    # ---- Time limits ----
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,

    # ---- Concurrency ----
    worker_concurrency=settings.celery_worker_concurrency,

    # ---- Result backend ----
    result_expires=86400,             # Results expire after 24h — prevents Redis bloat
    result_extended=True,             # Store task args/kwargs in result for debugging

    # ---- Connection resilience ----
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,

    # ---- Queue routing ----
    task_default_queue="default",
    task_queues={
        "default":     {"exchange": "default",     "routing_key": "default"},
        "rag":         {"exchange": "rag",          "routing_key": "rag"},
        "email":       {"exchange": "email",        "routing_key": "email"},
        "maintenance": {"exchange": "maintenance",  "routing_key": "maintenance"},
        "batch":       {"exchange": "batch",        "routing_key": "batch"},
        "webhooks":    {"exchange": "webhooks",     "routing_key": "webhooks"},
    },
    task_routes={
        "background_tasks.rebuild_rag_index":          {"queue": "rag"},
        "background_tasks.process_document_upload":    {"queue": "rag"},
        "background_tasks.apply_memory_decay":          {"queue": "maintenance"},
        "background_tasks.send_weekly_report":          {"queue": "email"},
        "background_tasks.send_weekly_reports_to_all":  {"queue": "maintenance"},
        "background_tasks.cleanup_expired_sessions":    {"queue": "maintenance"},
        "background_tasks.cleanup_old_usage_logs":      {"queue": "maintenance"},
        "background_tasks.cleanup_old_audit_logs":      {"queue": "maintenance"},
        "background_tasks.reset_monthly_token_counters": {"queue": "maintenance"},
        "background_tasks.process_batch_requests":      {"queue": "batch"},
        "background_tasks.deliver_webhook":             {"queue": "webhooks"},
    },

    # ---- Periodic tasks (Celery Beat schedule) ----
    beat_schedule={
        "cleanup-expired-sessions-daily": {
            "task": "background_tasks.cleanup_expired_sessions",
            "schedule": crontab(hour=3, minute=0),  # 3 AM UTC daily
        },
        "cleanup-old-usage-logs-weekly": {
            "task": "background_tasks.cleanup_old_usage_logs",
            "schedule": crontab(hour=4, minute=0, day_of_week=0),  # Sunday 4 AM UTC
            "kwargs": {"retention_days": 90},
        },
        "cleanup-old-audit-logs-monthly": {
            "task": "background_tasks.cleanup_old_audit_logs",
            "schedule": crontab(hour=5, minute=0, day_of_month=1),  # 1st of month
            "kwargs": {"retention_days": 365},
        },
        "send-weekly-reports": {
            "task": "background_tasks.send_weekly_reports_to_all",
            "schedule": crontab(hour=9, minute=0, day_of_week=1),  # Monday 9 AM UTC
        },
        "reset-monthly-token-counters": {
            "task": "background_tasks.reset_monthly_token_counters",
            "schedule": crontab(hour=0, minute=0, day_of_month=1),  # 1st of month, midnight
        },
        "apply-memory-decay-weekly": {
            "task": "background_tasks.apply_memory_decay",
            "schedule": crontab(hour=2, minute=30, day_of_week=0),  # Sunday 2:30 AM UTC
        },
    },
)


# ===========================================================================
# Signal Handlers
# ===========================================================================

from celery.signals import task_failure, task_success, worker_ready


@worker_ready.connect
def on_worker_ready(**kwargs):
    logger.info("Celery worker ready and listening for tasks")


@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    logger.error(
        "Celery task failed: task=%s id=%s error=%s",
        getattr(sender, "name", "unknown"), task_id, exception,
    )


@task_success.connect
def on_task_success(sender=None, **kwargs):
    logger.debug("Celery task succeeded: task=%s", getattr(sender, "name", "unknown"))


if __name__ == "__main__":
    celery_app.start()