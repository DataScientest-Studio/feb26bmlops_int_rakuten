import os
import requests
import streamlit as st

st.set_page_config(page_title="Live Predict | Rakuten MLOps", page_icon="🔮", layout="wide")

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("🔮 Live Predict")
st.caption("Classify a French product description using a trained LinearSVM model.")

# ── Inputs ─────────────────────────────────────────────────────────────────────
run_id = st.text_input(
    "Run ID",
    value=st.session_state.get("last_run_id", ""),
    placeholder="linearSVM_20260325_141817",
    help="The run ID returned after training (e.g. linearSVM_20260325_141817)",
)

text = st.text_area(
    "Product description (French)",
    placeholder="Saisissez la description du produit en français...",
    height=120,
)

predict_clicked = st.button("Predict", type="primary")

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_clicked:
    if not run_id.strip():
        st.warning("Please enter a Run ID. Train a model first on the Live Training page.")
    elif not text.strip():
        st.warning("Please enter a product description.")
    else:
        payload = {"run_id": run_id.strip(), "text": text.strip()}
        try:
            resp = requests.post(
                f"{API_URL}/predict/text/linear-svm",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            label_code = str(result.get("predicted_label", ""))
            confidence = result.get("confidence", 0.0)

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Predicted category:** `{label_code}`")
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
                st.progress(min(confidence, 1.0))

        except requests.ConnectionError:
            st.error(f"Cannot reach API at {API_URL}. Is the server running?")
        except requests.HTTPError as e:
            if e.response.status_code == 400:
                detail = e.response.json().get("detail", str(e))
                st.error(f"Prediction failed: {detail}. Check that the run_id is correct.")
            else:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
        except requests.Timeout:
            st.error("Request timed out.")
