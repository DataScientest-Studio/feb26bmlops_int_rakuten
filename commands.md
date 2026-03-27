- Running the API from Docker


- Running the API locally
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

- model training using the API endpint (Linear SVM)

curl -sS -X POST "http://localhost:8000/train/text/linear-svm"   -H "Content-Type: application/json"   -d '{}'

- model prediction using the API endpoint (Linear SVM)
curl -sS -X POST "http://localhost:8000/predict/text/linear-svm"   -H "Content-Type: application/json"   -d '{"run_id":"linearSVM_20260309_163508","text":"hermann hesse"}'


- running streamlit from streamlit_app directory

uv run streamlit run app.py  