import json
import os

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.transformers
import pandas as pd
from mlflow.tracking import MlflowClient
import dagshub
import shutil

LOCAL_BEST_DIR = "./models/latest"
os.makedirs(LOCAL_BEST_DIR, exist_ok=True)

BEST_MODEL_PATHS = {
    "SVM": os.path.join(LOCAL_BEST_DIR, "svm_latest.model"),
    "BERT": os.path.join(LOCAL_BEST_DIR, "bert_latest.model"),
    "RESNET": os.path.join(LOCAL_BEST_DIR, "resnet50_latest.model")
}

# # # 1. define mlflow experiment
repo_owner = os.getenv("DAGSHUB_USER", "andreiistudor")
repo_name = os.getenv("DAGSHUB_REPO", "feb26bmlops_int_rakuten")
token = os.getenv("DAGSHUB_TOKEN", None)

if token is None:
    dagshub.init(repo_owner="knanw", repo_name="feb26bmlops_int_rakuten", mlflow=True)
else:
    print(f"Using DagsHub token from environment variable: \n{repo_owner}\n{repo_name}\n{token}")
    # os.environ["DAGSHUB_USER_TOKEN"] = token
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

# dagshub.init(repo_owner="knanw", repo_name="feb26bmlops_int_rakuten", mlflow=True)

mlflow.set_experiment("Rakuten-Classification")


def train_and_log(model, model_name, X_test, y_test, metrics_path):
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


def evaluate_and_promote(new_metrics, model_name, metrics_path):
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
                filter_string=f"name='{model_name_registry}'")[0].version
            client.set_registered_model_alias(
                model_name_registry, alias, latest_version)

            update_local_best_file(model_name, metrics_path)
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
                client.set_registered_model_alias(model_name_registry, alias, first_version.version)

                update_local_best_file(model_name, metrics_path)

                print(f"Model {model_name_registry} version {first_version.version} is now Production.")
        except Exception as inner_e:
            print(f"Error. Model never registered. {inner_e}")


def log_image_training_run(
    model,
    model_name,
    session_folder,
    csv_log,
    final_model_path,
    use_transfer_learning,
    resume_path,
    step=None,
):
    """Log image training outputs to MLflow/DagsHub and return logged metrics."""
    run_name = f"IMAGE_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    registered_model_name = f"Rakuten_IMAGE_{str(model_name).upper()}"
    logged_metrics = {}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_family", "IMAGE")
        mlflow.log_param("image_model", model_name)
        mlflow.log_param("use_transfer_learning", bool(use_transfer_learning))
        mlflow.log_param("resume_used", bool(resume_path))
        if step is not None:
            mlflow.log_param("step", int(step))

        if csv_log and os.path.exists(csv_log):
            history = pd.read_csv(csv_log, sep=";")
            if not history.empty:
                latest = history.iloc[-1]
                metrics = {
                    "train_loss": float(latest.get("train_loss", 0.0)),
                    "val_loss": float(latest.get("val_loss", 0.0)),
                    "train_accuracy": float(latest.get("train_accuracy", 0.0)),
                    "val_accuracy": float(latest.get("val_accuracy", 0.0)),
                    "train_f1_score": float(latest.get("train_f1_score", 0.0)),
                    "val_f1_score": float(latest.get("val_f1_score", 0.0)),
                    "train_f1_macro": float(latest.get("train_f1_macro", 0.0)),
                    "eval_f1_macro": float(latest.get("val_f1_macro", 0.0)),
                }
                mlflow.log_metrics(metrics)
                logged_metrics = metrics

        if session_folder and os.path.isdir(session_folder):
            mlflow.log_artifacts(session_folder, artifact_path="model")
        elif final_model_path and os.path.exists(final_model_path):
            mlflow.log_artifact(final_model_path, artifact_path="model")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=registered_model_name,
        )

    return logged_metrics

def update_local_best_file(model_name, source_path):
    """Kopiert den Inhalt des aktuellen Trainings-Ordners in den 'latest' Pfad."""
    # Wir suchen den passenden Key (SVM, BERT oder RESNET)
    key = "SVM" if "SVM" in model_name.upper() else "BERT" if "BERT" in model_name.upper() else "RESNET"
    target = BEST_MODEL_PATHS.get(key)

    if target:
            source_dir = os.path.dirname(os.path.abspath(source_path))
            
            # Sicherheits-Check: Existiert der Quellordner überhaupt?
            if not os.path.exists(source_dir):
                print(f"FAILED: Source directory {source_dir} not found!")
                return

            # Falls das Ziel bereits existiert (als Datei oder Ordner), löschen
            if os.path.exists(target):
                if os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
            
            try:
                # Kopiert den kompletten Inhalt von source_dir in den neuen Ordner target
                shutil.copytree(source_dir, target)
                print(f"--- SUCCESS: {key} updated. Files are now in: {target} ---")
                
                # Kontrolle: Was ist im Ordner gelandet?
                print(f"Files copied: {os.listdir(target)}")
            except Exception as e:
                print(f"Error during copying: {e}")
