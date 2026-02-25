import json
import os
import inspect
from datetime import datetime

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.loaders import load_train_validation_csv
from src.features.text_preprocess import compute_class_weights, fit_and_apply_label_encoder


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_function = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_text_bert_from_csv(cfg):
    experiment_id = cfg.get("experiment_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_checkpoint = cfg.get("model_ckpt", "jhu-clsp/mmBERT-base")
    text_column = cfg.get("text_col", "text_stripped")
    label_column = cfg.get("label_col", "prdtypecode")

    output_dir = cfg.get("output_dir")
    if not output_dir:
        output_dir = f"models/text/{model_checkpoint.split('/')[-1]}_{experiment_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the full configuration for reproducibility
    full_cfg = dict(cfg)
    full_cfg["experiment_id"] = experiment_id
    full_cfg["output_dir"] = output_dir
    with open(os.path.join(output_dir, "cfg.json"), "w", encoding="utf-8") as cfg_file:
        json.dump(full_cfg, cfg_file, indent=2)

    
    train_csv_path = cfg.get("train_csv_path")
    validation_csv_path = cfg.get("validation_csv_path")
    if not train_csv_path or not validation_csv_path:
        raise ValueError("Both 'train_csv_path' and 'validation_csv_path' are required")

    train_df, validation_df = load_train_validation_csv(
        train_csv_path=train_csv_path,
        validation_csv_path=validation_csv_path,
        text_column=text_column,
        label_column=label_column,
        sample_number=cfg.get("sample_number"),
        seed=cfg.get("seed", 42),
    )

    train_df, validation_df, _, id_to_label, label_to_id = fit_and_apply_label_encoder(
        train_df=train_df,
        validation_df=validation_df,
        label_column=label_column,
    )

    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as label_file:
        json.dump({"id2label": id_to_label, "label2id": label_to_id}, label_file, indent=2)

    train_hf_dataset = Dataset.from_pandas(train_df[[text_column, "label"]].reset_index(drop=True))
    validation_hf_dataset = Dataset.from_pandas(
        validation_df[[text_column, "label"]].reset_index(drop=True)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch[text_column],
            truncation=cfg.get("truncation", True),
            padding=cfg.get("padding", False),
            max_length=cfg.get("max_length", 256),
        )
        encoded.pop("token_type_ids", None)
        return encoded

    train_tokenized = train_hf_dataset.map(tokenize_batch, batched=True, remove_columns=[text_column])
    validation_tokenized = validation_hf_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[text_column],
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    use_class_weights = cfg.get("use_class_weights", True)
    if use_class_weights:
        class_weights = compute_class_weights(
            train_df=train_df,
            encoded_label_column="label",
            method=cfg.get("class_weight_method", "inv_freq"),
            epsilon=cfg.get("class_weight_eps", 1e-6),
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(id_to_label),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1_macro": metric_f1.compute(
                predictions=predictions,
                references=labels,
                average="macro",
            )["f1"],
        }

    training_args_kwargs = {
        "output_dir": output_dir,
        "seed": cfg.get("seed", 42),
        "learning_rate": cfg.get("lr", 2e-5),
        "per_device_train_batch_size": cfg.get("batch_size", 16),
        "per_device_eval_batch_size": cfg.get("batch_size", 16),
        "num_train_epochs": cfg.get("epochs", 2),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "save_strategy": cfg.get("save_strategy", "epoch"),
        "load_best_model_at_end": cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": cfg.get("metric_for_best_model", "f1_macro"),
        "greater_is_better": cfg.get("greater_is_better", True),
        "logging_steps": cfg.get("logging_steps", 100),
        "report_to": cfg.get("report_to", "none"),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "linear"),
        "fp16": cfg.get("fp16", False),
        "bf16": cfg.get("bf16", False),
        "label_smoothing_factor": cfg.get("label_smoothing_factor", 0.0),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 1),
        "save_total_limit": cfg.get("save_total_limit", 2),
        "warmup_ratio": cfg.get("warmup_ratio", 0.06),
    }

    training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_signature:
        training_args_kwargs["evaluation_strategy"] = cfg.get("eval_strategy", "epoch")
    elif "eval_strategy" in training_args_signature:
        training_args_kwargs["eval_strategy"] = cfg.get("eval_strategy", "epoch")

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tokenized,
        "eval_dataset": validation_tokenized,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "class_weights": class_weights if use_class_weights else None,
    }

    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    train_output = trainer.train()
    evaluation_output = trainer.evaluate()

    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as eval_file:
        json.dump({key: float(value) for key, value in evaluation_output.items()}, eval_file, indent=2)

    train_metrics = train_output.metrics if hasattr(train_output, "metrics") else {}
    with open(os.path.join(output_dir, "train_metrics.json"), "w", encoding="utf-8") as train_file:
        json.dump({key: float(value) for key, value in train_metrics.items()}, train_file, indent=2)

    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    return {
        "output_dir": output_dir,
        "run_id": os.path.basename(output_dir),
        "train_samples_used": len(train_df),
        "validation_samples_used": len(validation_df),
        "eval_accuracy": evaluation_output.get("eval_accuracy"),
        "eval_f1_macro": evaluation_output.get("eval_f1_macro"),
        "eval_loss": evaluation_output.get("eval_loss"),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
    }