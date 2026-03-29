from pydantic import BaseModel, Field
from enum import Enum
import os

_DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@db:5432/dst_db"
)

class TrainTextRequest(BaseModel):
    step: int | None = None
    db_url: str = _DEFAULT_DB_URL
    model_ckpt: str = "jhu-clsp/mmBERT-base"
    sample_number: float | None = 0.05
    batch_size: int = 16
    epochs: int = 2
    lr: float = 2e-5


class TrainLinearSVMTextRequest(BaseModel):
    step: int | None = None
    db_url: str = _DEFAULT_DB_URL
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

###schemas for the Image Classification API

class ModelType(str, Enum):
    alexnet = "alexnet"
    vgg16 = "vgg16"
    resnet50 = "resnet50"


class FineTuneMode(str, Enum):
    classifier = "classifier"
    full = "full"
    resnet_selective = "resnet_selective"


class SchedulerType(str, Enum):
    steplr = "steplr"
    cosine = "cosine"
    plateau = "plateau"


class TrainRequest(BaseModel):
    model_type: ModelType = ModelType.resnet50
    mode: FineTuneMode = FineTuneMode.classifier
    epochs: int = Field(default=10, ge=1, le=500)
    lr_cls: float = Field(default=1e-2, gt=0)
    lr_back: float = Field(default=1e-3, gt=0)
    scheduler: SchedulerType = SchedulerType.steplr
    step_size: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.1, gt=0)
    resume: str | None = None
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    cm_every: int = Field(default=5, ge=1)


class TrainImageSyncRequest(TrainRequest):
    step: int | None = None
    use_transfer_learning: bool = True


class TrainImageSyncResponse(BaseModel):
    status: str
    final_model_path: str
    resume_used: str | None = None


class ImageDbResetRequest(BaseModel):
    output_folder: str = "data/image_db"


class ImageDbUpdateRequest(BaseModel):
    step: int
    db_url: str = _DEFAULT_DB_URL
    sample_number: float | None = None
    image_column: str = "image_file"
    label_column: str = "prdtypecode"
    input_folder: str = "data/image_data"
    output_folder: str = "data/image_db"


class ImageDbResponse(BaseModel):
    status: str
    output_folder: str
    step: int | None = None
    train_file_count: int = 0
    val_file_count: int = 0


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_epoch: int = 0
    total_epochs: int = 0
    last_train_loss: float | None = None
    last_val_loss: float | None = None
    last_val_accuracy: float | None = None
    last_val_f1: float | None = None
    session_folder: str | None = None
    error: str | None = None


class PredictResponse(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    top_k: list[dict] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str | None
    model_path: str | None
    device: str
    num_classes: int


class DbStatusResponse(BaseModel):
    status: str  # "connected" | "error"
    total_rows: int | None = None
    distinct_categories: int | None = None
    rows_per_step: dict | None = None
    detail: str | None = None