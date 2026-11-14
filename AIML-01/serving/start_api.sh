#!/bin/sh
set -e
if [ "$STORE" = "feast" ]; then
  (cd feature_repo && feast apply && feast materialize-incremental "$(date -u -d '+1 day' +%Y-%m-%dT%H:%M:%S)")
fi
exec uvicorn serving.api:app --host 0.0.0.0 --port 8000
