import os
import streamlit as st

st.set_page_config(page_title="Architecture | Rakuten MLOps", page_icon="🏗️", layout="wide")

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("🏗️ Architecture")

# ── System Overview ────────────────────────────────────────────────────────────
st.subheader("System Overview")
st.markdown("""
```
User
 └── Streamlit UI (port 8501)
       ├── FastAPI server (port 8000)
       │     ├── PostgreSQL DB (port 5432)   — product data
       │     └── MLflow / DagsHub            — experiment tracking
       └── Airflow (port 8080)
             └── FastAPI server              — triggers training DAGs
```
""")

st.divider()

# ── Services ──────────────────────────────────────────────────────────────────
st.subheader("Services")

import pandas as pd

services = pd.DataFrame([
    {"Service": "FastAPI",    "URL": "localhost:8000",  "Role": "Training & inference endpoints"},
    {"Service": "Airflow",    "URL": "localhost:8080",  "Role": "DAG orchestration (ingestion, training)"},
    {"Service": "PostgreSQL", "URL": "localhost:5432",  "Role": "Product data storage"},
    {"Service": "MLflow",     "URL": "DagsHub remote",  "Role": "Experiment tracking & model registry"},
    {"Service": "Streamlit",  "URL": "localhost:8501",  "Role": "This UI"},
])
st.dataframe(services, use_container_width=True, hide_index=True)

st.divider()

# ── API Endpoints ──────────────────────────────────────────────────────────────
st.subheader("API Endpoints")

endpoints = pd.DataFrame([
    {"Method": "GET",  "Path": "/health",                  "Input": "—",                                     "Output": "status, model_loaded, device"},
    {"Method": "POST", "Path": "/train/text/linear-svm",   "Input": "sample_number, c, max_iter, ngrams…",   "Output": "run_id, eval_accuracy, eval_f1_macro"},
    {"Method": "POST", "Path": "/train/text",              "Input": "sample_number, batch_size, epochs, lr", "Output": "run_id, eval_accuracy, eval_f1_macro"},
    {"Method": "POST", "Path": "/predict/text/linear-svm", "Input": "run_id, text",                          "Output": "predicted_label, confidence"},
    {"Method": "POST", "Path": "/predict/text",            "Input": "run_id, text, max_length",              "Output": "predicted_label, confidence"},
    {"Method": "POST", "Path": "/train",                   "Input": "model_type, epochs, lr_cls…",           "Output": "job_id"},
    {"Method": "GET",  "Path": "/jobs/{job_id}",           "Input": "job_id",                                "Output": "status, current_epoch, val_accuracy"},
    {"Method": "POST", "Path": "/predict",                 "Input": "image file",                            "Output": "class_name, confidence, top_k"},
])
st.dataframe(endpoints, use_container_width=True, hide_index=True)

st.divider()

# ── Live Health Check ──────────────────────────────────────────────────────────
st.subheader("Live API Health")

if st.button("Check API Health"):
    import requests
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        resp.raise_for_status()
        st.success("API is reachable")
        st.json(resp.json())
    except requests.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Is the server running?")
    except requests.HTTPError as e:
        st.error(f"API returned {e.response.status_code}")
