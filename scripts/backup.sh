#!/bin/bash
set -e
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
mkdir -p $BACKUP_DIR

PGPASSWORD=$POSTGRES_PASSWORD pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER $POSTGRES_DB > $BACKUP_DIR/postgres_$DATE.sql

curl -X PUT "http://$QDRANT_HOST:6333/collections/chat_memory/snapshots" \
  -H "api-key: $QDRANT_API_KEY" \
  -o $BACKUP_DIR/qdrant_$DATE.snapshot

cp /data/redis/dump.rdb $BACKUP_DIR/redis_$DATE.rdb 2>/dev/null || true

find $BACKUP_DIR -type f -mtime +30 -delete
echo "Backup completed: $DATE"