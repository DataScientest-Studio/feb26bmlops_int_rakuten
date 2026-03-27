#!/usr/bin/env bash
# up.sh — Start the stack using already-built images (no build step)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

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

# ── 0. Tear down + free ports ──────────────────────────────────────────────────
log "Stopping any running services..."
docker compose down --remove-orphans 2>/dev/null || true

# ── 1. PostgreSQL ──────────────────────────────────────────────────────────────
log "Starting PostgreSQL..."
docker compose up -d --wait db
log "PostgreSQL is healthy."

# ── 2. Airflow init (skipped if already initialised) ──────────────────────────
airflow_ready=$(docker compose exec -T db psql -U postgres -d dst_db -tAc \
  "SELECT 1 FROM information_schema.tables WHERE table_name='ab_user';" 2>/dev/null || true)
if [ "$airflow_ready" = "1" ]; then
  log "Airflow database already initialised, skipping."
else
  log "Initialising Airflow database..."
  if ! docker compose up --exit-code-from airflow-init airflow-init; then
    log "ERROR: Airflow database init failed. Check: docker compose logs airflow-init"
    exit 1
  fi
  log "Airflow database initialised."
fi

# ── 3. Airflow webserver + scheduler ──────────────────────────────────────────
log "Starting Airflow webserver and scheduler..."
docker compose up -d airflow-webserver airflow-scheduler

# ── 4. API ─────────────────────────────────────────────────────────────────────
log "Starting API..."
docker compose up -d api
wait_for_url "http://localhost:8000/health" "FastAPI API" 24

# ── 5. Streamlit ───────────────────────────────────────────────────────────────
log "Starting Streamlit..."
docker compose up -d streamlit
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
