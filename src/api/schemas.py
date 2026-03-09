from pydantic import BaseModel, Field


class TrainTextRequest(BaseModel):
    train_csv_path: str = "data/processed/train_fixed.csv"
    validation_csv_path: str = "data/processed/test_fixed.csv"
    model_ckpt: str = "jhu-clsp/mmBERT-base"
    sample_number: float | None = 0.05
    batch_size: int = 16
    epochs: int = 2
    lr: float = 2e-5


class TrainLinearSVMTextRequest(BaseModel):
    train_csv_path: str = "data/processed/train_fixed.csv"
    validation_csv_path: str = "data/processed/test_fixed.csv"
    sample_number: float = 1.0
    c: float = 2.0
    max_iter: int = 5000
    ngram_min: int = 3
    ngram_max: int = 5
    min_df: int = 2
    max_features: int = 150000


class TrainTextResponse(BaseModel):
    run_id: str
    output_dir: str
    eval_accuracy: float | None = None
    eval_f1_macro: float | None = None
    eval_loss: float | None = None


class PredictTextRequest(BaseModel):
    run_id: str
    text: str = Field(..., min_length=1)
    max_length: int = 256


class PredictLinearSVMTextRequest(BaseModel):
    run_id: str
    text: str = Field(..., min_length=1)


class PredictTextResponse(BaseModel):
    run_id: str
    predicted_class_id: int
    predicted_label: str
    confidence: float