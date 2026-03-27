import os
import joblib
from contextlib import asynccontextmanager
from datetime import datetime

from sklearn.pipeline import Pipeline
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text

from src.api.schemas import (
    PredictLinearSVMTextRequest,
    PredictTextRequest,
    PredictTextResponse,
    TrainLinearSVMTextRequest,
    TrainTextRequest,
    TrainTextResponse,
    HealthResponse,
    DbStatusResponse,
)
from src.pipelines.text_pipeline import train_text_bert_from_csv, train_text_linear_svm
from src.models.predict_model import predict_text, predict_text_linear_svm
from src.models.mlflow_utils import train_and_log, evaluate_and_promote
from src.models.classifier import classifier_service
from . import train, predict, jobs

from dotenv import load_dotenv
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifier_service.load()
    except Exception as e:
        print(f"[WARNING] Could not load classifier model: {e}")
        print("[WARNING] /predict endpoints will return 503 until model is available.")
    yield


app = FastAPI(title="Rakuten Text API", version="0.1.0", lifespan=lifespan)

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


@app.get("/db/status", response_model=DbStatusResponse, tags=["health"])
async def db_status():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/dst_db")
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM product")).scalar()
            categories = conn.execute(text("SELECT COUNT(DISTINCT prdtypecode) FROM product")).scalar()
            rows = conn.execute(
                text("SELECT step, COUNT(*) as cnt FROM product GROUP BY step ORDER BY step")
            ).fetchall()
        return DbStatusResponse(
            status="connected",
            total_rows=total,
            distinct_categories=categories,
            rows_per_step={str(r[0] if r[0] is not None else "NULL"): r[1] for r in rows},
        )
    except Exception as exc:
        return DbStatusResponse(status="error", detail=str(exc))


@app.post("/train/text", response_model=TrainTextResponse)
def train_text_endpoint(request: TrainTextRequest):
    try:
        cfg = {"experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"), **request.model_dump()}
        output = train_text_bert_from_csv(cfg)
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
        return PredictTextResponse(**predict_text(request.model_dump()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train/text/linear-svm", response_model=TrainTextResponse)
def train_linear_svm_endpoint(request: TrainLinearSVMTextRequest):
    try:
        cfg = {"experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"), **request.model_dump()}
        output = train_text_linear_svm(cfg)

        vectorizer = joblib.load(os.path.join(output["output_dir"], "vectorizer.joblib"))
        model      = joblib.load(os.path.join(output["output_dir"], "linear_svm.joblib"))
        train_and_log(
            model=Pipeline([("tfidf", vectorizer), ("svm", model)]),
            model_name="SVM",
            X_test=None,
            y_test=None,
            metrics_path=os.path.join(output["output_dir"], "eval_metrics.json"),
        )
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
def predict_linear_svm_endpoint(request: PredictLinearSVMTextRequest):
    try:
        return PredictTextResponse(**predict_text_linear_svm(request.model_dump()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc