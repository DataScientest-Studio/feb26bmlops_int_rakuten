from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.utils.dates import days_ago
from datetime import datetime
import requests
import sqlalchemy
import logging

# Datenbank Verbindung
DB_CONN = "postgresql://postgres:postgres@db:5432/dst_db"

def decide_strategy():
    engine = sqlalchemy.create_engine(DB_CONN)
    count = engine.execute("SELECT count(*) FROM product WHERE step IS NULL").scalar()
    logging.info(f"--- found {count} table entries without step. ---")
    
    if count == 0 or count is None:
        logging.info("No new data found: jumping to single_train_text.")
        return "single_train_text"
    
    if count > 5000:
        limit = count // 10
        for i in range(1, 11):
            engine.execute(f"""
                UPDATE product SET step = {i} WHERE id IN (
                    SELECT id FROM product WHERE step IS NULL LIMIT {limit}
                )
            """)
        logging.info("--- New data found. Slicing data in 10 steps. ---")
        return "initial_training_group.train_text_step_1"
    else:
        engine.execute("UPDATE product SET step = 1 WHERE step IS NULL")
        logging.info("--- Sufficient new data found for single step retraining. ---")
        return "single_train_text"

def call_text_training_api(step, **kwargs):
    """
    Ruft die Training-API auf. 
    STANDARD: 'svm' (schont 4GB VRAM/GPU Probleme)
    """
    # 1. Modell-Typ aus Trigger-Konfiguration holen, Standard jetzt 'svm'
    dag_run_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
    model_type = dag_run_conf.get('model_type', 'svm') # <--- Geändert auf svm
    
    # 2. Default Endpunkte definieren
    default_endpoints = {
        "bert": "http://host.docker.internal:8000/train/text",
        "svm": "http://host.docker.internal:8000/train/text/linear-svm"
    }
    
    # 3. Hybrid-URL
    var_name = f"rakuten_url_{model_type}"
    target_url = Variable.get(var_name, default_var=default_endpoints.get(model_type))
    
    # 4. Batch-Größe festlegen (SVM = 50, BERT = 1)
    batch_size = 50 if model_type == 'svm' else 1
    
    payload = {
        "step": step,
        "batch_size": batch_size
    }
    
    logging.info(f"--- Calling API: {target_url} | Model: {model_type} | Step: {step} ---")
    
    try:
        # Timeout auf 600s, da SVM Training bei großen Batches Zeit braucht
        resp = requests.post(target_url, json=payload, timeout=1800) #1800 sec = 30 min
        if resp.status_code != 200:
            logging.error(f"API Error ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Request failed: {str(e)}")
        raise

def call_image_training_api(step, **kwargs):
    """
    Calls the Image Classification Training API.
    Currently supports: resnet50, vgg16, vit_b_16, alexnet
    """
    dag_run_conf = kwargs.get('dag_run').conf if kwargs.get('dag_run') else {}
    model_type = dag_run_conf.get('image_model', 'resnet50')  # Default ResNet50
    
    # Image training endpoint
    var_name = f"rakuten_url_image_{model_type}"
    default_image_endpoint = f"http://host.docker.internal:8000/train"
    target_url = Variable.get(var_name, default_var=default_image_endpoint)
    
    payload = {
        "model_type": model_type,
        "mode": "classifier",  # or "resnet_selective", "full"
        "epochs": 1,  # Adjust based on your needs
        "lr_cls": 0.01,
        "lr_back": 0.001,
        "scheduler": "steplr",
        "dropout": 0.2,
    }
    
    logging.info(f"--- Calling Image API: {target_url} | Model: {model_type} ---")
    
    try:
        resp = requests.post(target_url, json=payload, timeout=3600)  # 1 hour timeout
        if resp.status_code != 202:  # Image training returns 202 (async)
            logging.error(f"Image API Error ({resp.status_code}): {resp.text}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Image training request failed: {str(e)}")
        raise

with DAG('rakuten_training', 
         params={"model_type": "svm"},
         start_date=days_ago(2),
         schedule_interval=None, 
         catchup=False) as dag:

    branch = BranchPythonOperator(
        task_id='decide_strategy', 
        python_callable=decide_strategy
    )
    
    skip_training = EmptyOperator(task_id='skip_training')

    # PFAD B: Einzelschritt - Text Training
    single_train_text = PythonOperator(
        task_id='single_train_text',
        python_callable=call_text_training_api,
        op_kwargs={'step': 1}
    )

    # PFAD B: Einzelschritt - Image Training
    single_train_image = PythonOperator(
        task_id='single_train_image',
        python_callable=call_image_training_api,
        op_kwargs={'step': 1}
    )

    # PFAD A: 10 Schritte - Text + Image pairs
    with TaskGroup(group_id='initial_training_group') as training_group:
        previous_task = None
        for i in range(1, 11):
            # Text training for step i
            text_task = PythonOperator(
                task_id=f'train_text_step_{i}',
                python_callable=call_text_training_api,
                op_kwargs={'step': i}
            )
            
            # Image training for step i
            image_task = PythonOperator(
                task_id=f'train_image_step_{i}',
                python_callable=call_image_training_api,
                op_kwargs={'step': i}
            )
            
            # Chain: text >> image within each step
            text_task >> image_task
            
            # Chain to previous step's image
            if previous_task:
                previous_task >> text_task
            previous_task = image_task

    branch >> [training_group, single_train_text, skip_training]
    
    # Single path: text training → image training
    single_train_text >> single_train_image