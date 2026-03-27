import os
import requests
import streamlit as st

st.set_page_config(page_title="Live Training | Rakuten MLOps", page_icon="🚀", layout="wide")

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
AIRFLOW_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_AUTH = (
    os.getenv("AIRFLOW_USER", "airflow"),
    os.getenv("AIRFLOW_PASS", "airflow"),
)

st.title("🚀 Live Training")

tab_api, tab_airflow = st.tabs(["Direct API Training", "Trigger via Airflow DAG"])

# ── Tab 1: Direct API Training ─────────────────────────────────────────────────
with tab_api:
    st.subheader("Train LinearSVM")
    st.caption("Trains a TF-IDF + LinearSVC model directly via the API. Results are returned synchronously.")

    with st.form("svm_train_form"):
        col1, col2 = st.columns(2)

        with col1:
            sample_number = st.slider(
                "Data fraction", min_value=0.05, max_value=1.0, value=1.0, step=0.05,
                help="Fraction of the dataset to use for training (1.0 = full dataset)"
            )
            c = st.number_input(
                "C (regularisation)", min_value=0.01, value=2.0, step=0.5,
                help="LinearSVC regularisation parameter"
            )
            max_iter = st.number_input(
                "Max iterations", min_value=100, value=5000, step=500
            )

        with col2:
            ngram_min = st.number_input("N-gram min", min_value=1, value=3)
            ngram_max = st.number_input("N-gram max", min_value=1, value=5)
            min_df = st.number_input(
                "Min document frequency", min_value=1, value=2,
                help="Minimum number of documents a token must appear in"
            )
            max_features = st.number_input(
                "Max features", min_value=1000, value=150000, step=10000
            )

        submitted = st.form_submit_button("Train SVM", type="primary")

    if submitted:
        payload = {
            "sample_number": sample_number,
            "c": c,
            "max_iter": int(max_iter),
            "ngram_min": int(ngram_min),
            "ngram_max": int(ngram_max),
            "min_df": int(min_df),
            "max_features": int(max_features),
        }
        with st.spinner("Training in progress... (this may take a few minutes)"):
            try:
                resp = requests.post(
                    f"{API_URL}/train/text/linear-svm",
                    json=payload,
                    timeout=1800,
                )
                resp.raise_for_status()
                result = resp.json()

                run_id = result.get("run_id", "")
                st.session_state["last_run_id"] = run_id
                st.session_state["last_metrics"] = result

                st.success("Training complete!")
                st.markdown("**Run ID** (copy this for the Predict page):")
                st.code(run_id)

                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{result.get('eval_accuracy', 0):.3f}")
                m2.metric("F1 macro", f"{result.get('eval_f1_macro', 0):.3f}")
                if result.get("eval_loss") is not None:
                    m3.metric("Loss", f"{result['eval_loss']:.4f}")

            except requests.ConnectionError:
                st.error(f"Cannot reach API at {API_URL}. Is the server running?")
            except requests.Timeout:
                st.error("Training timed out after 30 minutes.")
            except requests.HTTPError as e:
                detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                st.error(f"Training failed ({e.response.status_code}): {detail}")

# ── Tab 2: Airflow DAG ─────────────────────────────────────────────────────────
with tab_airflow:
    st.subheader("Trigger Airflow Training DAG")
    st.caption(
        "Fires the `rakuten_training` DAG asynchronously. "
        "The DAG calls the API internally in up to 10 incremental steps. "
        "Run ID will be available in the Airflow task logs after completion."
    )

    model_type = st.selectbox("Model type", ["svm"], index=0)

    if st.button("Trigger DAG", type="primary"):
        try:
            resp = requests.post(
                f"{AIRFLOW_URL}/api/v1/dags/rakuten_training/dagRuns",
                json={"conf": {"model_type": model_type}},
                auth=AIRFLOW_AUTH,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            dag_run_id = data.get("dag_run_id", "")
            st.success(f"DAG triggered successfully.")
            st.markdown("**DAG Run ID:**")
            st.code(dag_run_id)
            st.session_state["last_dag_run_id"] = dag_run_id
        except requests.ConnectionError:
            st.error(f"Cannot reach Airflow at {AIRFLOW_URL}.")
        except requests.HTTPError as e:
            detail = e.response.text if e.response else str(e)
            st.error(f"Airflow returned {e.response.status_code}: {detail}")

    st.divider()
    st.subheader("DAG Run Status")

    dag_run_id_input = st.text_input(
        "DAG Run ID",
        value=st.session_state.get("last_dag_run_id", ""),
        placeholder="manual__2026-03-26T...",
    )

    if st.button("Refresh Status"):
        if not dag_run_id_input.strip():
            st.warning("Enter a DAG Run ID first.")
        else:
            try:
                resp = requests.get(
                    f"{AIRFLOW_URL}/api/v1/dags/rakuten_training/dagRuns/{dag_run_id_input.strip()}",
                    auth=AIRFLOW_AUTH,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                state = data.get("state", "unknown")
                colour = {"success": "green", "running": "blue", "failed": "red"}.get(state, "gray")
                st.markdown(f"State: :{colour}[**{state.upper()}**]")
                with st.expander("Full response"):
                    st.json(data)
            except requests.ConnectionError:
                st.error(f"Cannot reach Airflow at {AIRFLOW_URL}.")
            except requests.HTTPError as e:
                st.error(f"Airflow returned {e.response.status_code}: {e.response.text}")

    st.divider()
    st.subheader("Recent DAG Runs")

    if st.button("Load recent runs"):
        try:
            resp = requests.get(
                f"{AIRFLOW_URL}/api/v1/dags/rakuten_training/dagRuns"
                "?limit=5&order_by=-start_date",
                auth=AIRFLOW_AUTH,
                timeout=10,
            )
            resp.raise_for_status()
            runs = resp.json().get("dag_runs", [])
            if runs:
                import pandas as pd
                df = pd.DataFrame([
                    {
                        "Run ID": r.get("dag_run_id", ""),
                        "State": r.get("state", ""),
                        "Start": r.get("start_date", ""),
                        "End": r.get("end_date", ""),
                    }
                    for r in runs
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No DAG runs found.")
        except requests.ConnectionError:
            st.error(f"Cannot reach Airflow at {AIRFLOW_URL}.")
        except requests.HTTPError as e:
            st.error(f"Airflow returned {e.response.status_code}: {e.response.text}")
