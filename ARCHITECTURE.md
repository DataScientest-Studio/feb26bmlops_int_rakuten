# Architecture Overview — Text Classification Pipeline

> Scope: BERT and LinearSVM text models.
> Image pipeline is implemented but not covered here.

---

## What This Project Does

Given a product title/description, classify it into one of 27 product categories (Rakuten France dataset).
Two models are supported:

| Model | Approach |
|-------|----------|
| **mmBERT** | Fine-tuned multilingual BERT via HuggingFace Transformers |
| **LinearSVM** | TF-IDF character n-grams + scikit-learn LinearSVC |

Both models are trained and served through a **FastAPI** HTTP API backed by a **PostgreSQL** database.
Experiments are tracked with **MLflow** (hosted on DagsHub). Data is versioned with **DVC**.

---

## End-to-End Data Flow

```
                        ┌─────────────────────────────────────┐
                        │          PostgreSQL (docker)         │
                        │  table: products                     │
                        │  columns: text_stripped, prdtypecode │
                        └──────────────┬──────────────────────┘
                                       │ SQL query (80/20 split)
                                       ▼
                              src/data/loaders.py
                              (train_df, val_df)
                                       │
                                       ▼
                          src/features/text_preprocess.py
                          - LabelEncoder (27 classes → 0..26)
                          - class weights (inv_freq)
                                       │
                                       ▼
              ┌────────────────────────┴────────────────────────┐
              │                                                  │
      BERT branch                                        SVM branch
              │                                                  │
  AutoTokenizer + AutoModel                      TfidfVectorizer (char ngrams)
  WeightedTrainer (HuggingFace)                  LinearSVC (balanced weights)
              │                                                  │
              └────────────────────────┬────────────────────────┘
                                       │
                              models/text/{run_id}/
                              ├── model weights
                              ├── label_map.json
                              ├── eval_metrics.json
                              └── cfg.json
                                       │
                                       ▼
                              src/models/mlflow_utils.py
                              - logs metrics + artifacts
                              - promotes best model → "production"
```

---

## File-by-File Guide

### API Layer (`src/api/`)

#### `app.py` — Entry point
Starts the FastAPI application. Defines four routes for the text pipeline:

| Method | Route | Action |
|--------|-------|--------|
| POST | `/train/text` | Train a BERT model |
| POST | `/train/text/linear-svm` | Train a LinearSVM model |
| POST | `/predict/text` | Predict with a saved BERT model |
| POST | `/predict/text/linear-svm` | Predict with a saved LinearSVM model |

Also exposes `GET /health` for status checks.

#### `schemas.py` — Request and response shapes
Pydantic models that validate what comes in and goes out of each endpoint.
This is also where **most default hyperparameter values are declared** (see section below).

Key schemas:
- `TrainTextRequest` — BERT training parameters (batch size, epochs, learning rate, …)
- `TrainLinearSVMTextRequest` — SVM parameters (C, max_iter, ngram range, …)
- `PredictTextRequest` — run_id + text to classify
- `TrainTextResponse` / `PredictTextResponse` — returned results


---

### Model Layer (`src/models/`)

#### `predict_model.py` — Inference
Two functions, one per model type:

- `predict_text(cfg)` — loads tokenizer + BERT model from disk, runs forward pass, returns label + confidence
- `predict_text_linear_svm(cfg)` — loads `vectorizer.joblib` + `linear_svm.joblib`, applies softmax to decision scores

Both functions resolve the model directory from a `run_id` (timestamp string like `20240315_143022`).

#### `mlflow_utils.py` — Experiment tracking
- Logs all metrics and model artifacts to MLflow (DagsHub remote)
- Compares the new model against the current "production" champion on `eval_f1_macro`
- Promotes the winner by setting the `production` alias in the Model Registry

---

### Pipeline Layer (`src/pipelines/`)

#### `text_pipeline.py` — Where the actual training happens
This is the core of the project. Two functions, each opening with a **single parameter block** that lists every setting with its default in one place (no scattered `cfg.get()` calls throughout the body).

**`train_text_bert_from_csv(cfg)`**
1. Load data via `loaders.py` (SQL, 80/20 split)
2. Encode labels and compute class weights via `text_preprocess.py`
3. Tokenize with `AutoTokenizer` (default: `jhu-clsp/mmBERT-base`)
4. Fine-tune with a custom `WeightedTrainer` (subclass of HuggingFace `Trainer`)
5. Save model, tokenizer, `label_map.json`, metrics, config to `models/text/{run_id}/`

**`train_text_linear_svm_from_csv(cfg)`**
1. Same data loading and label encoding
2. Fit `TfidfVectorizer` with character n-grams
3. Fit `LinearSVC` with balanced class weights
4. Save `vectorizer.joblib`, `linear_svm.joblib`, metrics, config

---

### Data & Features (`src/data/`, `src/features/`)

#### `loaders.py` — Data loading
Two modes, same output:
- `load_train_validation_sql()` — queries PostgreSQL via SQLAlchemy, returns `(train_df, val_df)` with an 80/20 split
- `load_train_validation_csv()` — reads from CSV files instead

Both accept a `sample_number` fraction (0–1] to work with a subset of data during development.

#### `text_preprocess.py` — Label encoding and class weights
- `fit_and_apply_label_encoder()` — fits a `LabelEncoder` on the training labels, applies it to both splits. Adds an `encoded_label` column.
- `compute_class_weights()` — computes per-class weights to handle imbalanced data. Two methods:
  - `inv_freq` — weight ∝ 1/frequency (default)
  - `sqrt_inv` — smoother variant: weight ∝ 1/√frequency

---

## Hardcoded Values — Where They Cluster

Values typed directly in source code rather than read from environment or config files.

### Database connection
Appears in 2 places with the same literal string:

| File | Value |
|------|-------|
| `src/api/schemas.py` | `"postgresql://postgres:postgres@localhost:5432/dst_db"` |
| `tests/test_sql_loader.py` | same |

`docker-compose.yml` declares matching `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`.

### Column names
| File | Value |
|------|-------|
| `src/pipelines/text_pipeline.py` | `text_col = "text_stripped"`, `label_col = "prdtypecode"` |
| `tests/test_sql_loader.py` | same |

### BERT training defaults
Split between `schemas.py` (user-facing, exposed via HTTP) and the parameter block at the top of `train_text_bert_from_csv` in `text_pipeline.py` (internal):

| Parameter | Value | Exposed in API? |
|-----------|-------|-----------------|
| model checkpoint | `"jhu-clsp/mmBERT-base"` | yes (schemas.py) |
| batch size | 16 | yes |
| epochs | 2 | yes |
| learning rate | 2e-5 | yes |
| max token length | 256 | yes |
| seed | 42 | no (text_pipeline.py) |
| weight decay | 0.01 | no |
| warmup ratio | 0.06 | no |
| lr scheduler | `"linear"` | no |
| label smoothing | 0.0 | no |
| fp16 / bf16 | False | no |
| logging steps | 100 | no |
| gradient accumulation | 1 | no |

### SVM training defaults
Same split: user-facing in `schemas.py`, internal in the parameter block of `train_text_linear_svm_from_csv`.

| Parameter | Value | Exposed in API? |
|-----------|-------|-----------------|
| C | 2.0 | yes (schemas.py) |
| max_iter | 5000 | yes |
| ngram_min / ngram_max | 3 / 5 | yes |
| min_df | 2 | yes |
| max_features | 150 000 | yes |
| analyzer | `"char"` | no (text_pipeline.py) |
| class_weight | `"balanced"` | no |

### MLflow / DagsHub
All in `src/models/mlflow_utils.py`:

| Setting | Value |
|---------|-------|
| DagsHub owner | `"knanw"` |
| DagsHub repo | `"feb26bmlops_int_rakuten"` |
| Experiment name | `"Rakuten-Text-Classification"` |
| Champion alias | `"production"` |
| Comparison metric | `"eval_f1_macro"` |
| BERT registry name | `"Rakuten_BERT"` |
| SVM registry name | `"Rakuten_SVM"` |

---

## Dependency Map (simplified)

```
app.py
  ├── text_pipeline.py
  │     ├── loaders.py
  │     └── text_preprocess.py
  ├── predict_model.py
  └── mlflow_utils.py  (SVM train endpoint only)
```

---

## Key Tools & Libraries

| Tool | Role |
|------|------|
| **FastAPI** | HTTP API framework |
| **Pydantic** | Request/response validation and defaults |
| **HuggingFace Transformers** | BERT tokenizer, model, and Trainer |
| **scikit-learn** | TfidfVectorizer, LinearSVC, LabelEncoder |
| **SQLAlchemy + psycopg2** | PostgreSQL access |
| **MLflow + DagsHub** | Experiment tracking, model registry |
| **DVC** | Data versioning (S3 remote) |
| **Docker Compose** | Local PostgreSQL instance |
| **GitHub Actions** | CI: lint (flake8) + test (pytest) on push to master |
| **uv** | Fast Python package manager (replaces pip) |
