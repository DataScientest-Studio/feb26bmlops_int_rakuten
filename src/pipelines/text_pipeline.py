import json
import os
import inspect
from datetime import datetime

import evaluate
import joblib
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.loaders import load_train_validation_sql
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
    # --- all parameters in one place ---
    # user-facing (defaults must match TrainTextRequest in schemas.py)
    step          = cfg.get("step")
    db_url        = cfg.get("db_url")
    sample_number = cfg.get("sample_number")
    model_ckpt    = cfg.get("model_ckpt", "jhu-clsp/mmBERT-base")
    batch_size    = cfg.get("batch_size", 16)
    epochs        = cfg.get("epochs", 2)
    lr            = cfg.get("lr", 2e-5)
    max_length    = cfg.get("max_length", 256)
    # internal — not exposed via the API
    experiment_id           = cfg.get("experiment_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    text_col                = cfg.get("text_col", "text_stripped")
    label_col               = cfg.get("label_col", "prdtypecode")
    seed                    = cfg.get("seed", 42)
    truncation              = cfg.get("truncation", True)
    padding                 = cfg.get("padding", False)
    use_class_weights       = cfg.get("use_class_weights", True)
    class_weight_method     = cfg.get("class_weight_method", "inv_freq")
    class_weight_eps        = cfg.get("class_weight_eps", 1e-6)
    eval_strategy           = cfg.get("eval_strategy", "epoch")
    save_strategy           = cfg.get("save_strategy", "epoch")
    save_total_limit        = cfg.get("save_total_limit", 2)
    load_best_model_at_end  = cfg.get("load_best_model_at_end", True)
    metric_for_best_model   = cfg.get("metric_for_best_model", "f1_macro")
    greater_is_better       = cfg.get("greater_is_better", True)
    logging_steps           = cfg.get("logging_steps", 100)
    report_to               = cfg.get("report_to", "none")
    lr_scheduler_type       = cfg.get("lr_scheduler_type", "linear")
    fp16                    = cfg.get("fp16", False)
    bf16                    = cfg.get("bf16", False)
    label_smoothing_factor  = cfg.get("label_smoothing_factor", 0.0)
    warmup_ratio            = cfg.get("warmup_ratio", 0.06)
    weight_decay            = cfg.get("weight_decay", 0.01)
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)

    if not db_url:
        raise ValueError("'db_url' is required")

    output_dir = cfg.get("output_dir") or f"models/text/{model_ckpt.split('/')[-1]}_{experiment_id}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg, "experiment_id": experiment_id, "output_dir": output_dir}, f, indent=2)

    train_df, validation_df = load_train_validation_sql(
        db_url=db_url, step=step, text_column=text_col,
        label_column=label_col, sample_number=sample_number, seed=seed,
    )

    train_df, validation_df, _, id_to_label, label_to_id = fit_and_apply_label_encoder(
        train_df=train_df, validation_df=validation_df, label_column=label_col,
    )

    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": id_to_label, "label2id": label_to_id}, f, indent=2)

    train_hf_dataset = Dataset.from_pandas(train_df[[text_col, "label"]].reset_index(drop=True))
    validation_hf_dataset = Dataset.from_pandas(validation_df[[text_col, "label"]].reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)

    def tokenize_batch(batch):
        encoded = tokenizer(batch[text_col], truncation=truncation, padding=padding, max_length=max_length)
        encoded.pop("token_type_ids", None)
        return encoded

    train_tokenized = train_hf_dataset.map(tokenize_batch, batched=True, remove_columns=[text_col])
    validation_tokenized = validation_hf_dataset.map(tokenize_batch, batched=True, remove_columns=[text_col])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(
            train_df=train_df, encoded_label_column="label",
            method=class_weight_method, epsilon=class_weight_eps,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=len(id_to_label), id2label=id_to_label, label2id=label_to_id,
    )

    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1_macro": metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        }

    training_args_kwargs = {
        "output_dir": output_dir,
        "seed": seed,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "logging_steps": logging_steps,
        "report_to": report_to,
        "lr_scheduler_type": lr_scheduler_type,
        "fp16": fp16,
        "bf16": bf16,
        "label_smoothing_factor": label_smoothing_factor,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_total_limit": save_total_limit,
        "warmup_ratio": warmup_ratio,
    }

    training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_signature:
        training_args_kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in training_args_signature:
        training_args_kwargs["eval_strategy"] = eval_strategy

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

    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in evaluation_output.items()}, f, indent=2)

    train_metrics = train_output.metrics if hasattr(train_output, "metrics") else {}
    with open(os.path.join(output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in train_metrics.items()}, f, indent=2)

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


def train_text_linear_svm(cfg):
    # --- all parameters in one place ---
    # user-facing (defaults must match TrainLinearSVMTextRequest in schemas.py)
    step          = cfg.get("step")
    db_url        = cfg.get("db_url")
    sample_number = cfg.get("sample_number")
    c             = cfg.get("c", 2.0)
    max_iter      = cfg.get("max_iter", 5000)
    ngram_min     = cfg.get("ngram_min", 3)
    ngram_max     = cfg.get("ngram_max", 5)
    min_df        = cfg.get("min_df", 2)
    max_features  = cfg.get("max_features", 150000)
    # internal — not exposed via the API
    experiment_id = cfg.get("experiment_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    text_col      = cfg.get("text_col", "text_stripped")
    label_col     = cfg.get("label_col", "prdtypecode")
    seed          = cfg.get("seed", 42)
    class_weight  = cfg.get("class_weight", "balanced")
    analyzer      = cfg.get("analyzer", "char")

    if not db_url:
        raise ValueError("'db_url' is required")

    output_dir = cfg.get("output_dir") or f"models/text/linearSVM_{experiment_id}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "cfg.json"), "w", encoding="utf-8") as f:
        json.dump({**cfg, "experiment_id": experiment_id, "output_dir": output_dir}, f, indent=2)

    train_df, validation_df = load_train_validation_sql(
        db_url=db_url, step=step, text_column=text_col,
        label_column=label_col, sample_number=sample_number, seed=seed,
    )

    train_df, validation_df, _, id_to_label, label_to_id = fit_and_apply_label_encoder(
        train_df=train_df, validation_df=validation_df, label_column=label_col,
    )

    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": id_to_label, "label2id": label_to_id}, f, indent=2)

    print(f"Training LinearSVM: C={c}, max_iter={max_iter}, ngram=({ngram_min},{ngram_max}), "
          f"min_df={min_df}, max_features={max_features}, analyzer={analyzer}")

    vectorizer = TfidfVectorizer(
        ngram_range=(ngram_min, ngram_max), min_df=min_df,
        max_features=max_features, analyzer=analyzer,
    )
    x_train = vectorizer.fit_transform(train_df[text_col].astype(str).tolist())
    x_validation = vectorizer.transform(validation_df[text_col].astype(str).tolist())

    model = LinearSVC(C=c, class_weight=class_weight, max_iter=max_iter)
    model.fit(x_train, train_df["label"].values)

    print("Training completed. Evaluating on validation set...")

    preds = model.predict(x_validation)
    eval_accuracy = float(accuracy_score(validation_df["label"].values, preds))
    eval_f1_macro = float(f1_score(validation_df["label"].values, preds, average="macro"))
    eval_metrics  = {"eval_accuracy": eval_accuracy, "eval_f1_macro": eval_f1_macro}

    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.joblib"))
    joblib.dump(model, os.path.join(output_dir, "linear_svm.joblib"))


    return {
        "output_dir": output_dir,
        "run_id": os.path.basename(output_dir),
        "train_samples_used": len(train_df),
        "validation_samples_used": len(validation_df),
        "eval_accuracy": eval_accuracy,
        "eval_f1_macro": eval_f1_macro,
        "eval_loss": None,
    }