# Rakuten Multimodal Classification (Text + Image + Weighted Fusion)

ML pipeline architecture and API design for a multimodal classification task combining text and image data.

## Goal

Provide a clean architecture where:

- weighted fusion (text + image) is callable from code/API
- prediction is exposed through the same API


## Timeline

- **Phase 0 — Kick-off**: before **Feb 13, 2026**
- **Phase 1 — Foundations**: deadline **Feb 26, 2026**
- **Phase 2 — Microservices, Tracking & Versioning**: deadline **Mar 10, 2026**
- **Phase 4 — Monitoring & Automation**: deadline **Mar 25, 2026**
- **Final defense**: **Mar 30, 2026**

### Phase 1 (Foundations) expected deliverables

- define objectives and key metrics
- reproducible development environment
- data collection and preprocessing pipeline
- one-time data loading script into SQL/NoSQL storage
- baseline training and evaluation
- training and prediction Python scripts
- basic inference API with training and prediction endpoints



## Reference links

- Kick-off slides: https://docs.google.com/presentation/d/1qIjvUaZMwHl6vvmQfJErMwcwCs4BbyHja6v9Lk4RBWQ/edit#slide=id.g31a2ff139c2_0_0
- Sprint planning: https://docs.google.com/spreadsheets/d/1yq9xkqW25IZz9dAmfDA5ZM9f3zldd2UYTSTXGDZNr7w/edit?gid=1397140348#gid=1397140348
- Studio repository: https://github.com/DataScientest-Studio/feb26bmlops_int_rakuten
- Defense guidelines: https://docs.google.com/document/d/1bF9K4yBjaeWvBRdnNCIpwHDLqdZUHX1VRiEpQOQPY0A/edit?tab=t.0
- Archive (slides/streamlit): https://drive.google.com/drive/folders/1q3fFLqENeoFD66BD6UP5eIJYcnJsag23?usp=drive_link

## Current status

## Package setup (recommended)

Use editable install so imports from `src` work from scripts outside `src/`.

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Then run training with the simplest command (no `-m`):

```bash
python3 scripts/run_train.py
```

Alternative (module mode):

```bash
python3 -m src.models.train_model
```

## Target structure

```text
feb26bmlops_int_rakuten/
├── AGENTS.md
├── README.md
├── scripts/
│   └── text_pipeline/
│       └── ... (research notebooks)
├── src/
│   ├── config/
│   │   ├── settings.py
│   │   └── paths.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── loaders.py
│   ├── features/
│   │   ├── text_preprocess.py
│   │   └── image_preprocess.py
│   ├── pipelines/
│   │   ├── text_pipeline.py
│   │   ├── image_pipeline.py
│   │   └── fusion_pipeline.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   ├── registry.py
│   │   └── artifacts.py
│   └── api/
│       ├── app.py
│       ├── schemas.py
│       └── services.py
└── models/
    ├── text/
    ├── image/
    └── fusion/
```

## Train and predict API (minimal)

Implemented endpoints:

- `POST /train/text`
- `POST /predict/text`
- `GET /health`

### Run API locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Example requests

Train:

```bash
curl -X POST http://localhost:8000/train/text \
    -H "Content-Type: application/json" \
    -d '{
        "train_csv_path": "data/processed/train_fixed.csv",
        "validation_csv_path": "data/processed/test_fixed.csv",
        "model_ckpt": "jhu-clsp/mmBERT-base",
        "sample_number": 0.05,
        "batch_size": 16,
        "epochs": 2,
        "lr": 0.00002
    }'
```

Predict:

```bash
curl -X POST http://localhost:8000/predict/text \
    -H "Content-Type: application/json" \
    -d '{
        "run_id": "mmBERT-base_YYYYMMDD_HHMMSS",
        "text": "Votre texte produit ici",
        "max_length": 256
    }'
```


## Migration plan (notebooks -> src)

1. Move reusable text preprocessing + training logic from notebooks into `src/pipelines/text_pipeline.py`.
2. Add image preprocessing + training in `src/pipelines/image_pipeline.py`.
3. Add fusion logic (weighted probabilities) in `src/pipelines/fusion_pipeline.py`.
4. Wire `src/models/train_model.py` and `src/models/predict_model.py` as reusable entry points.
5. Expose these entry points through `src/api/app.py`.

## Next implementation step

The next coding step is to scaffold `src/pipelines/` and `src/api/`, then implement a first operational version of:

- `train_text()`
- `train_image()`
- `train_fusion()`
- `predict_multimodal()`

all routed through FastAPI service functions.
