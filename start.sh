#!/usr/bin/env bash
# start.sh — Start the full Rakuten MLOps stack
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 0. Tear down any existing stack + free required ports ─────────────────────
log "Stopping any running services and removing volumes for a fresh start..."
docker compose down --volumes --remove-orphans 2>/dev/null || true

log "Freeing required ports (5432, 8000, 8080, 8501)..."
for port in 5432 8000 8080 8501; do
  containers=$(docker ps --filter "publish=${port}" --format "{{.ID}}" 2>/dev/null || true)
  if [ -n "$containers" ]; then
    log "  Stopping container(s) on port ${port}: $(docker ps --filter "publish=${port}" --format "{{.Names}}" | tr '\n' ' ')"
    echo "$containers" | xargs docker stop > /dev/null
  fi
done
log "All services stopped."

wait_for_url() {
  local url="$1" label="$2" max="${3:-30}"
  log "Waiting for $label ..."
  for i in $(seq 1 "$max"); do
    if curl -sf "$url" > /dev/null 2>&1; then
      log "$label is up."
      return 0
    fi
    sleep 5
  done
  log "ERROR: $label did not become healthy after $((max * 5))s"
  exit 1
}

# ── 1. PostgreSQL ──────────────────────────────────────────────────────────────
log "Starting PostgreSQL..."
docker compose up -d --wait --force-recreate db
log "PostgreSQL is healthy."

# ── 2. Airflow init (one-time job, blocks until complete) ──────────────────────
log "Initialising Airflow database (this may take a while on first run)..."
if ! docker compose up --exit-code-from airflow-init airflow-init; then
  log "ERROR: Airflow database init failed. Check logs: docker compose logs airflow-init"
  exit 1
fi
log "Airflow database initialised."

# ── 3. Airflow webserver + scheduler ──────────────────────────────────────────
log "Starting Airflow webserver and scheduler..."
docker compose up -d airflow-webserver airflow-scheduler

# ── 4. API ─────────────────────────────────────────────────────────────────────
log "Building and starting API..."
docker compose up -d --build --force-recreate api
wait_for_url "http://localhost:8000/health" "FastAPI API" 24

# ── 5. Streamlit ───────────────────────────────────────────────────────────────
log "Building and starting Streamlit..."
docker compose up -d --build --force-recreate streamlit
wait_for_url "http://localhost:8501/_stcore/health" "Streamlit" 24

# ── Summary ────────────────────────────────────────────────────────────────────
log "=================================================="
log "  Stack is running:"
log "    Streamlit UI  ->  http://localhost:8501"
log "    FastAPI docs  ->  http://localhost:8000/docs"
log "    Airflow UI    ->  http://localhost:8080  (airflow / airflow)"
log "    PostgreSQL    ->  localhost:5432          (postgres / postgres)"
log "    MLflow        ->  https://dagshub.com/knanw/feb26bmlops_int_rakuten.mlflow"
log "=================================================="
