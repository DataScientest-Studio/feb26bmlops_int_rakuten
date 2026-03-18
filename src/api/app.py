import os
import joblib

from sklearn.pipeline import Pipeline
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

from src.models.mlflow_utils import (
    train_and_log, 
    evaluate_and_promote)

from contextlib import asynccontextmanager
from . import train, predict, jobs
from src.models.classifier import classifier_service
from src.api.schemas import HealthResponse

from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    try:
        classifier_service.load()
    except Exception as e:
        print(f"[WARNING] Could not load classifier model: {e}")
        print("[WARNING] /predict endpoints will return 503 until model is available.")
    yield
    # Shutdown: nothing to clean up

app = FastAPI(title="Rakuten Text API", version="0.1.0", lifespan=lifespan)


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


        # MLflow Tracking
        vectorizer = joblib.load(
            os.path.join(output["output_dir"], "vectorizer.joblib")
        )
        model = joblib.load(os.path.join(output["output_dir"], "linear_svm.joblib"))

        text_clf = Pipeline([("tfidf", vectorizer), ("svm", model)])

        train_and_log(
            model=text_clf,
            model_name="SVM",
            X_test=None,
            y_test=None,
            metrics_path=os.path.join(output["output_dir"], "eval_metrics.json"),
        )

        # Champion-Comparison
        evaluate_and_promote(new_metrics=output, model_name="SVM")


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



app.include_router(predict.router)
app.include_router(train.router)
app.include_router(jobs.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=classifier_service.is_loaded(),
        model_type=classifier_service.model_type,
        model_path=classifier_service.model_path,
        device=classifier_service.device,
        num_classes=classifier_service.num_classes,
    )