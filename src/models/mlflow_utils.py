import json
import os

import mlflow
import mlflow.sklearn
import mlflow.transformers
import pandas as pd
from mlflow.tracking import MlflowClient
import dagshub

_dagshub_initialized = False


def _init_mlflow():
    """Initialize DagsHub + MLflow once, only when a token is available."""
    global _dagshub_initialized
    if _dagshub_initialized:
        return
    token = os.getenv("DAGSHUB_USER_TOKEN", "")
    if token:
        dagshub.init(repo_owner="knanw", repo_name="feb26bmlops_int_rakuten", mlflow=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Rakuten-Text-Classification")
    _dagshub_initialized = True


def train_and_log(model, model_name, X_test, y_test, metrics_path):
    _init_mlflow()
    with mlflow.start_run(
        run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d')}"
    ):
        # load metrics from eval_metrics.json
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            metrics = {k: round(v, 4) for k, v in metrics.items()}

        # performance tracking
        mlflow.log_metrics(metrics)

        # parameter tracking(e.g. model type)
        mlflow.log_param("model_family", model_name)

        # saving artifacts (entire folder models/text/...)
        mlflow.log_artifacts(os.path.dirname(metrics_path), artifact_path="model")

        # register model officially (important for registry!)
        if "SVM" in model_name:
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="Rakuten_SVM"
            )
        else:
            mlflow.transformers.log_model(
                model, "model", registered_model_name="Rakuten_BERT"
            )


def evaluate_and_promote(new_metrics, model_name):
    _init_mlflow()
    client = MlflowClient()
    model_name_registry = f"Rakuten_{model_name}"
    alias = "production"

    try:
        # 1. fetch current "Production" model
        model_version_details = client.get_model_version_by_alias(
            model_name_registry, alias
        )

        if not model_version_details:
            raise IndexError("No Production model found")

        # 2. get metrics from previous model
        old_run = client.get_run(model_version_details.run_id)
        old_f1 = old_run.data.metrics.get("eval_f1_macro", 0)

        # 3. comparison
        new_f1 = round(new_metrics.get("eval_f1_macro", 0), 4)

        if new_f1 > old_f1:
            print(f"New model ({new_f1}) is better than old ({old_f1})!")
            # mark new version as production
            latest_version = client.search_model_versions(
                filter_string=f"name='{model_name_registry}'"
            )[0].version
            client.set_registered_model_alias(
                model_name_registry, alias, latest_version
            )
        else:
            print(f"Old model remains champion (Old F1: {old_f1}, New F1: {new_f1}).")

    except Exception as e:
        print(f"Initial setup or Error: {e}")
        print("Marking the current model as the first Production champion.")
        try:
            all_versions = client.search_model_versions(
                filter_string=f"name='{model_name_registry}'"
            )
            if all_versions:
                first_version = max(all_versions, key=lambda v: int(v.version))
                client.set_registered_model_alias(
                    model_name_registry, alias, first_version.version
                )
                print(
                    f"Model {model_name_registry} version {first_version.version} is now Production."
                )
        except Exception as inner_e:
            print(f"Error. Model never registered. {inner_e}")