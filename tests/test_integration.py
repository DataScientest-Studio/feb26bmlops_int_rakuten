"""
Integration tests — require all Docker services to be running.

Run from project root:
    pytest tests/test_integration.py -v

Override base URLs via env vars:
    API_BASE_URL      (default: http://localhost:8000)
    AIRFLOW_BASE_URL  (default: http://localhost:8080)
    AIRFLOW_USER      (default: airflow)
    AIRFLOW_PASS      (default: airflow)
"""

import os
import pytest
import requests

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
AIRFLOW_BASE = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "airflow")

TIMEOUT = 10


# ── API ──────────────────────────────────────────────────────────────────────

def test_api_reachable():
    """API container is up and returns a valid response on /health."""
    r = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


def test_api_health_schema():
    """Health endpoint returns the expected JSON keys."""
    r = requests.get(f"{API_BASE}/health", timeout=TIMEOUT)
    data = r.json()
    for key in ("status", "model_loaded", "device", "num_classes"):
        assert key in data, f"Missing key '{key}' in /health response"
    assert data["status"] == "ok"


# ── Database ─────────────────────────────────────────────────────────────────

def test_db_connected():
    """DB status endpoint reports a live PostgreSQL connection."""
    r = requests.get(f"{API_BASE}/db/status", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "connected", (
        f"DB not connected. Detail: {data.get('detail')}"
    )


def test_db_loaded():
    """Product table contains at least one row (SQL init file was loaded)."""
    r = requests.get(f"{API_BASE}/db/status", timeout=TIMEOUT)
    data = r.json()
    assert data.get("status") == "connected", "DB not connected — skipping row check"
    total = data.get("total_rows", 0)
    assert total > 0, (
        f"Product table is empty (total_rows={total}). "
        "Check that import_daten.sql was loaded by the db container."
    )


def test_db_categories():
    """Product table contains at least one distinct category (27 expected for full dataset)."""
    import warnings
    r = requests.get(f"{API_BASE}/db/status", timeout=TIMEOUT)
    data = r.json()
    if data.get("status") != "connected":
        pytest.skip("DB not connected")
    cats = data.get("distinct_categories", 0)
    assert cats >= 1, f"No categories found in product table (distinct_categories={cats})"
    if cats != 27:
        warnings.warn(f"Expected 27 categories for full dataset but found {cats}", UserWarning)


# ── Airflow ───────────────────────────────────────────────────────────────────

def test_airflow_reachable():
    """Airflow webserver is up and returns a valid health response."""
    r = requests.get(
        f"{AIRFLOW_BASE}/health",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


def test_airflow_metadb_healthy():
    """Airflow metadatabase reports healthy."""
    r = requests.get(
        f"{AIRFLOW_BASE}/health",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        timeout=TIMEOUT,
    )
    data = r.json()
    status = data.get("metadatabase", {}).get("status", "unknown")
    assert status == "healthy", f"Airflow metadatabase status: {status}"


def test_airflow_scheduler_healthy():
    """Airflow scheduler reports healthy."""
    r = requests.get(
        f"{AIRFLOW_BASE}/health",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        timeout=TIMEOUT,
    )
    data = r.json()
    status = data.get("scheduler", {}).get("status", "unknown")
    assert status == "healthy", f"Airflow scheduler status: {status}"


def test_airflow_training_dag_exists():
    """The rakuten_training DAG is registered in Airflow."""
    r = requests.get(
        f"{AIRFLOW_BASE}/api/v1/dags/rakuten_training",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, (
        f"DAG 'rakuten_training' not found (status {r.status_code}). "
        "Check that the dags/ volume is mounted correctly."
    )


# ── Streamlit ─────────────────────────────────────────────────────────────────

def test_streamlit_reachable():
    """Streamlit container is up and serving the app."""
    streamlit_base = os.getenv("STREAMLIT_BASE_URL", "http://localhost:8501")
    r = requests.get(streamlit_base, timeout=TIMEOUT)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
