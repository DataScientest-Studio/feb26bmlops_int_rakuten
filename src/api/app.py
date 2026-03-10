from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    PredictLinearSVMTextRequest,
    PredictTextRequest,
    PredictTextResponse,
    TrainLinearSVMTextRequest,
    TrainTextRequest,
    TrainTextResponse,
)
from src.api.services import (
    predict_text_linear_svm_service,
    predict_text_service,
    train_text_linear_svm_service,
    train_text_service,
)

app = FastAPI(title="Rakuten Text API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/train/text", response_model=TrainTextResponse)
def train_text_endpoint(request: TrainTextRequest):
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
def predict_text_endpoint(request: PredictTextRequest):
    try:
        output = predict_text_service(request.model_dump())
        return PredictTextResponse(**output)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train/text/linear-svm", response_model=TrainTextResponse)
def train_text_linear_svm_endpoint(request: TrainLinearSVMTextRequest):
    try:
        output = train_text_linear_svm_service(request.model_dump())
        return TrainTextResponse(
            run_id=output["run_id"],
            output_dir=output["output_dir"],
            eval_accuracy=output.get("eval_accuracy"),
            eval_f1_macro=output.get("eval_f1_macro"),
            eval_loss=output.get("eval_loss"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/text/linear-svm", response_model=PredictTextResponse)
def predict_text_linear_svm_endpoint(request: PredictLinearSVMTextRequest):
    try:
        output = predict_text_linear_svm_service(request.model_dump())
        return PredictTextResponse(**output)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc