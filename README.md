# Rakuten Multimodal Classification (Text + Image + Weighted Fusion)

ML pipeline architecture and API design for a multimodal classification task combining text and image data.

## Goal

Provide a clean architecture where:

- weighted fusion (text + image) is callable from code/API
- prediction is exposed through the same API


## Reference links

- Kick-off slides: https://docs.google.com/presentation/d/1qIjvUaZMwHl6vvmQfJErMwcwCs4BbyHja6v9Lk4RBWQ/edit#slide=id.g31a2ff139c2_0_0
- Sprint planning: https://docs.google.com/spreadsheets/d/1yq9xkqW25IZz9dAmfDA5ZM9f3zldd2UYTSTXGDZNr7w/edit?gid=1397140348#gid=1397140348
- Studio repository: https://github.com/DataScientest-Studio/feb26bmlops_int_rakuten
- Defense guidelines: https://docs.google.com/document/d/1bF9K4yBjaeWvBRdnNCIpwHDLqdZUHX1VRiEpQOQPY0A/edit?tab=t.0
- Archive (slides/streamlit): https://drive.google.com/drive/folders/1q3fFLqENeoFD66BD6UP5eIJYcnJsag23?usp=drive_link

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

## Train and predict API


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

```bashTrainingArguments
curl -X POST http://localhost:8000/predict/text \
    -H "Content-Type: application/json" \
    -d '{
        "run_id": "mmBERT-base_YYYYMMDD_HHMMSS",
        "text": "Votre texte produit ici",
        "max_length": 256
    }'
```

### Image classification API

The image API exposes synchronous and asynchronous training, single-image inference, batch inference, and helper endpoints used by Airflow to rebuild `data/image_db`.

Image train (synchronous, used by Airflow):

```bash
curl -X POST http://localhost:8000/train/sync \
    -H "Content-Type: application/json" \
    -d '{
        "model_type": "resnet50",
        "mode": "classifier",
        "epochs": 1,
        "lr_cls": 0.01,
        "lr_back": 0.001,
        "scheduler": "steplr",
        "dropout": 0.2,
        "step": 1,
        "use_transfer_learning": false
    }'
```

Image train (asynchronous, returns a job id immediately):

```bash
curl -X POST http://localhost:8000/train \
    -H "Content-Type: application/json" \
    -d '{
        "model_type": "resnet50",
        "mode": "classifier",
        "epochs": 3,
        "lr_cls": 0.01,
        "lr_back": 0.001,
        "scheduler": "steplr",
        "dropout": 0.2,
        "resume": "./models/image/resnet50_latest.model"
    }'
```

The async endpoint returns a `job_id`. Training progress can then be checked through the jobs endpoint:

```bash
curl http://localhost:8000/jobs/<job_id>
```

Single-image predict:

```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \
    -H "accept: application/json" \
    -F "file=@/absolute/path/to/image.jpg"
```

Batch predict:

```bash
curl -X POST http://localhost:8000/predict/batch \
    -H "accept: application/json" \
    -F "files=@/absolute/path/to/image1.jpg" \
    -F "files=@/absolute/path/to/image2.jpg"
```


# DVC/DagsHub

```ShellScript
dvc remote list
#if dagshub-project is visible -> ok
```

### create dagshub .dv/config.local file:

```ShellScript
dvc remote modify origin --local auth basic
```

### configure dagshub-project (doesn't come with git-project)

```ShellScript
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USER
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
```

To use the dockerize api version first call:
```ShellScript
make dagshub_token=YOUR_DAGSHUB_TOKEN dagshub_user=YOUR_DAGSHUB_USER
```
This will set the needed data in the .env.local file needed by docker compose

### Download data

```ShellScript
dvc status
dvc pull
```

### dvc track, commit changes to dagshub \*.sql file:

```ShellScript
dvc add data/import_daten.sql
git add data/import_daten.sql.dvc
git commit -m "fix: restore dvc tracking for sql data"
dvc push
git push origin master
```

### Build postgres container

```ShellScript
docker compose up -d --build
```

### delete all docker project files (e.g. container, network, volume)

```ShellScript
docker compose down -v
```

### Logging, troubleshooting container

```ShellScript
docker logs pg_container
docker exec -it pg_container psql -U postgres -d dst_db
```

### DB connection String

#### Syntax: postgresql://username:password@localhost/dbname

```ShellScript
engine = create_engine('postgresql://postgres:postgres@localhost:5432/dst_db')
```

### DagsHub-URL verknüpfen (falls nicht in .dvc/config)

ggf. dvc init
dvc remote add -d origin https://dagshub.com/knanw/feb26bmlops_int_rakuten.dvc

### for local MLFlow integration

#### 1. start mlflow docker container

```ShellScript
#docker run -d -p 5000:5000 --name mlflow_server ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0
```

#### 2. replace dagshub.init to mlflow.set_tracking_uri

```ShellScript
# dagshub.init(repo_owner="knanw", repo_name="feb26bmlops_int_rakuten", mlflow=True)
mlflow.set_tracking_uri("http://localhost:5000")
```
