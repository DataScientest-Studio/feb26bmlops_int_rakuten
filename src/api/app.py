from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    PredictTextRequest,
    PredictTextResponse,
    TrainTextRequest,
    TrainTextResponse,
)
from src.api.services import predict_text_service, train_text_service

app = FastAPI(title="Rakuten Text API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/train/text", response_model=TrainTextResponse)
def train_text_endpoint(request):
    try:
        output = train_text_service(request.model_dump())
        return TrainTextResponse(
            run_id=output["run_id"],
            output_dir=output["output_dir"],
            eval_accuracy=output.get("eval_accuracy"),
            eval_f1_macro=output.get("eval_f1_macro"),
            eval_loss=output.get("eval_loss"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/text", response_model=PredictTextResponse)
def predict_text_endpoint(request):
    try:
        output = predict_text_service(request.model_dump())
        return PredictTextResponse(**output)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc