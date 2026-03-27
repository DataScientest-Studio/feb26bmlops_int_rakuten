import streamlit as st

st.set_page_config(
    page_title="Rakuten MLOps",
    page_icon="🛒",
    layout="wide",
)

# Initialize shared session state keys used across pages
if "last_run_id" not in st.session_state:
    st.session_state["last_run_id"] = ""
if "last_metrics" not in st.session_state:
    st.session_state["last_metrics"] = {}

st.title("Rakuten MLOps Dashboard")
st.markdown("Use the sidebar to navigate between pages.")
