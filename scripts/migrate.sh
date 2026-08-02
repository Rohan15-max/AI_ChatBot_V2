#!/bin/bash
set -e
ENV=${1:-production}
if [ "$ENV" = "production" ]; then
    alembic upgrade head
else
    alembic upgrade head --sql
fi