- Running the API from Docker

docker run --rm -p 8000:8000 rakuten-api-test

- Running the API locally
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload


- Testing the API with curl
curl -X POST "http://localhost:8000/train_text" -H "Content-Type: application/json" -d '{"text_col": "text_stripped", "experiment_id": "20240229_123456"}'