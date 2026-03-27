import os
import requests
import streamlit as st

st.set_page_config(page_title="Home | Rakuten MLOps", page_icon="🏠", layout="wide")

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
AIRFLOW_BASE = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "airflow")


def _get(url, timeout=5, **kwargs):
    try:
        r = requests.get(url, timeout=timeout, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def check_api():
    data, err = _get(f"{API_BASE}/health")
    if err:
        return {"ok": False, "error": err}
    return {"ok": True, **data}


def check_db():
    data, err = _get(f"{API_BASE}/db/status")
    if err:
        return {"ok": False, "error": err}
    return {"ok": data.get("status") == "connected", **data}


def check_airflow():
    data, err = _get(
        f"{AIRFLOW_BASE}/health",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
    )
    if err:
        return {"ok": False, "error": err}
    metadb = data.get("metadatabase", {}).get("status", "unknown")
    scheduler = data.get("scheduler", {}).get("status", "unknown")
    return {
        "ok": metadb == "healthy" and scheduler == "healthy",
        "metadb": metadb,
        "scheduler": scheduler,
    }


def status_badge(ok: bool) -> str:
    return "🟢 Healthy" if ok else "🔴 Unhealthy"


# ── Header ──────────────────────────────────────────────────────────────────
st.title("Rakuten Product Classification — MLOps Dashboard")
st.markdown(
    "Live interface for the **Rakuten France product classification** system. "
    "Products are classified into **27 categories** based on French-language descriptions."
)

col1, col2, col3 = st.columns(3)
col1.metric("Categories", "27", help="Rakuten France product taxonomy")
col2.metric("Model", "LinearSVM", help="TF-IDF + LinearSVC text classifier")
col3.metric("Language", "French", help="Product descriptions are in French")

st.divider()

# ── Infrastructure Health ────────────────────────────────────────────────────
st.subheader("Infrastructure Health")

refresh = st.button("Refresh status")

if refresh or "infra_status" not in st.session_state:
    with st.spinner("Checking services…"):
        st.session_state["infra_status"] = {
            "api": check_api(),
            "db": check_db(),
            "airflow": check_airflow(),
        }

s = st.session_state["infra_status"]
api_s = s["api"]
db_s = s["db"]
af_s = s["airflow"]

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**FastAPI**")
    st.write(status_badge(api_s["ok"]))
    if api_s["ok"]:
        model_status = "loaded" if api_s.get("model_loaded") else "not loaded"
        st.caption(f"Model: {model_status}")
        if api_s.get("model_type"):
            st.caption(f"Type: {api_s['model_type']}")
    else:
        st.caption(f"Error: {api_s.get('error', 'unknown')}")

with c2:
    st.markdown("**PostgreSQL**")
    st.write(status_badge(db_s["ok"]))
    if db_s["ok"]:
        total = db_s.get("total_rows")
        st.caption(f"Rows: {total:,}" if isinstance(total, int) else "Rows: ?")
        st.caption(f"Categories: {db_s.get('distinct_categories', '?')}")
        rows_per_step = db_s.get("rows_per_step") or {}
        if rows_per_step:
            steps_summary = ", ".join(
                f"step {k}: {v:,}" for k, v in sorted(rows_per_step.items())
            )
            st.caption(f"Distribution — {steps_summary}")
    else:
        st.caption(f"Error: {db_s.get('error', db_s.get('detail', 'unknown'))}")

with c3:
    st.markdown("**Airflow**")
    st.write(status_badge(af_s["ok"]))
    if "metadb" in af_s:
        st.caption(f"Metadb: {af_s['metadb']}")
        st.caption(f"Scheduler: {af_s['scheduler']}")
    else:
        st.caption(f"Error: {af_s.get('error', 'unknown')}")

st.divider()

# ── Project Objectives ───────────────────────────────────────────────────────
st.subheader("Project Objectives")
st.markdown("""
- **Automated categorisation** — classify any French product description into the correct Rakuten category
- **Incremental training** — retrain on new data without reprocessing the full dataset
- **Experiment tracking** — every training run is logged in MLflow (DagsHub)
- **Orchestration** — Airflow DAGs manage data ingestion and training pipelines
- **This UI** — live training, live prediction, and pipeline monitoring from a single interface
""")

st.divider()

st.subheader("Quick Links")
col1, col2, col3 = st.columns(3)
col1.link_button("FastAPI docs", f"{API_BASE}/docs")
col2.link_button("Airflow UI", "http://localhost:8080")
col3.link_button("MLflow (DagsHub)", "https://dagshub.com/knanw/feb26bmlops_int_rakuten.mlflow")
