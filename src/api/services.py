from datetime import datetime

from src.models.predict_model import predict_text
from src.models.train_model import train_model


def train_text_service(payload: dict) -> dict:
    cfg = {
        "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "text_col": "text_stripped",
        "label_col": "prdtypecode",
        "model_type": "text",
        "seed": 42,
        "model_ckpt": payload.get("model_ckpt", "jhu-clsp/mmBERT-base"),
        "train_csv_path": payload.get("train_csv_path", "data/processed/train_fixed.csv"),
        "validation_csv_path": payload.get("validation_csv_path", "data/processed/test_fixed.csv"),
        "sample_number": payload.get("sample_number", 0.05),
        "use_class_weights": True,
        "class_weight_method": "inv_freq",
        "class_weight_eps": 1e-6,
        "max_length": payload.get("max_length", 256),
        "padding": False,
        "truncation": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 100,
        "report_to": "none",
        "lr_scheduler_type": "linear",
        "fp16": False,
        "bf16": False,
        "label_smoothing_factor": 0.0,
        "warmup_ratio": 0.06,
        "batch_size": payload.get("batch_size", 16),
        "lr": payload.get("lr", 2e-5),
        "epochs": payload.get("epochs", 2),
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
    }
    return train_model(cfg)


def predict_text_service(payload: dict) -> dict:
    return predict_text(payload)