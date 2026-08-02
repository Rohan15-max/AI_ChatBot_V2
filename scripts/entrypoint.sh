#!/bin/bash
set -e
python -c "from database import init_db; init_db(); print('Database tables ready.')"
exec "$@"