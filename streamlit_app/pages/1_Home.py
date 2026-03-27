import streamlit as st

st.set_page_config(page_title="Home | Rakuten MLOps", page_icon="🏠", layout="wide")

st.title("🛒 Rakuten Product Classification — MLOps Dashboard")
st.markdown("""
This dashboard provides a live interface for the **Rakuten France product classification** system.
The model automatically categorises product listings into one of **27 categories** based on their
French-language title and description.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Categories", "27", help="Rakuten France product taxonomy")
col2.metric("Model", "LinearSVM", help="TF-IDF + LinearSVC text classifier")
col3.metric("Language", "French", help="Product descriptions are in French")

st.divider()

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
col1.link_button("FastAPI docs", "http://localhost:8000/docs")
col2.link_button("Airflow UI", "http://localhost:8080")
col3.link_button("MLflow (DagsHub)", "https://dagshub.com/knanw/feb26bmlops_int_rakuten.mlflow")
