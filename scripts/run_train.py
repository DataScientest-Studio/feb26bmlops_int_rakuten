import json
from datetime import datetime

from src.models.train_model import train_model


def build_default_cfg():
    return {
        "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "text_col": "text_stripped",
        "label_col": "prdtypecode",
        "model_type": "text",
        "seed": 42,
        "model_ckpt": "jhu-clsp/mmBERT-base",
        "train_csv_path": "data/processed/train_fixed.csv",
        "validation_csv_path": "data/processed/test_fixed.csv",
        "sample_number": 0.05,
        "use_class_weights": True,
        "class_weight_method": "inv_freq",
        "class_weight_eps": 1e-6,
        "max_length": 256,
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
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 2,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
    }


if __name__ == "__main__":
    cfg = build_default_cfg()
    output = train_model(cfg)
    print(json.dumps(output, indent=2))
