import json
from datetime import datetime

from src.pipelines.text_pipeline import train_text_bert_from_csv


def train_text_model(cfg):
    return train_text_bert_from_csv(cfg)


def train_image_model(cfg):
    raise NotImplementedError("Image training placeholder: to be implemented in a future step")


def train_fusion_model(cfg):
    raise NotImplementedError("Fusion training placeholder: to be implemented in a future step")


def train_model(cfg):
    model_type = cfg.get("model_type", "text")

    if model_type == "text":
        return train_text_model(cfg)

    if model_type == "image":
        return train_image_model(cfg)

    if model_type == "fusion":
        return train_fusion_model(cfg)

    raise ValueError("model_type must be one of: text, image, fusion")


if __name__ == "__main__":
    CFG = {
        # experiment setup
        "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "text_col": "text_stripped",
        "label_col": "prdtypecode",
        "model_type": "text",
        "seed": 42,
        
        # model
        "model_ckpt": "jhu-clsp/mmBERT-base",
        
        # data paths
        "train_csv_path": "data/processed/train_fixed.csv",
        "validation_csv_path": "data/processed/test_fixed.csv",
        "sample_number": 0.05,
        
        # imbalance handling
        "use_class_weights": True,
        "class_weight_method": "inv_freq",
        "class_weight_eps": 1e-6,
        
        # tokenizer
        "max_length": 256,
        "padding": False,
        "truncation": True,
        
        # model training
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

    output = train_model(CFG)
    print(json.dumps(output, indent=2))
