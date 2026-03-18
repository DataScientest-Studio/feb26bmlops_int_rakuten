import json
import os

import dagshub
import joblib
import mlflow

# Konfiguration für DagsHub (falls noch nicht global gesetzt)
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'Dein_Nutzername'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'Dein_Token'
# mlflow.set_tracking_uri('https://dagshub.com/Nutzer/Repo.mlflow')


def log_existing_models(base_path="models/text"):
    dagshub.init(repo_owner="knanw", repo_name="feb26bmlops_int_rakuten", mlflow=True)

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Bestimme den Modell-Typ (BERT oder SVM) aus dem Ordnernamen
        model_type = "BERT" if "BERT" in folder_name else "SVM"

        with mlflow.start_run(run_name=folder_name):
            # 1. Metriken aus JSON laden und tracken
            metrics_path = os.path.join(folder_path, "eval_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                    # Erwartet: {"eval_accuracy": 0.9, "eval_f1_macro": 0.85}
                    metrics = {k: round(v, 4) for k, v in metrics.items()}
                    mlflow.log_metrics(metrics)

            # 2. Parameter loggen (z.B. Modelltyp)
            mlflow.log_param("model_family", model_type)

            # 3. Modell-Dateien als Artefakte speichern
            # Wir laden einfach den kompletten Unterordner hoch
            mlflow.log_artifacts(folder_path, artifact_path="model")

            # 4. Speziell für SVM (als offizielles MLflow-Model registrieren)
            if model_type == "SVM":
                svm_file = os.path.join(folder_path, "linear_svm")  # oder joblib Datei
                if os.path.exists(svm_file):
                    # Lädt das Modell so, dass DagsHub ein "Deploy"-Button anzeigen kann
                    model = joblib.load(svm_file)
                    mlflow.sklearn.log_model(
                        model, "model", registered_model_name="Rakuten_SVM"
                    )


if __name__ == "__main__":
    log_existing_models()
