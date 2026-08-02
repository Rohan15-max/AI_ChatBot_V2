#!/bin/bash
set -e
BACKUP_DIR="/backups"
LATEST_POSTGRES=$(ls -t $BACKUP_DIR/postgres_*.sql | head -1)
LATEST_QDRANT=$(ls -t $BACKUP_DIR/qdrant_*.snapshot | head -1)
LATEST_REDIS=$(ls -t $BACKUP_DIR/redis_*.rdb | head -1)

echo "Restoring PostgreSQL from $LATEST_POSTGRES"
PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB < $LATEST_POSTGRES

echo "Restoring Qdrant from $LATEST_QDRANT"
curl -X PUT "http://$QDRANT_HOST:6333/collections/chat_memory/snapshots/recover" \
  -H "api-key: $QDRANT_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"snapshot_path\": \"$LATEST_QDRANT\"}"

echo "Restoring Redis (requires restart)"
cp $LATEST_REDIS /data/redis/dump.rdb
docker-compose restart redis
echo "Restore completed."