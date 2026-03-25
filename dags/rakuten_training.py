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
        logging.info("No new data found: jumping to skip_training.")
        return "skip_training"
    
    if count > 5000:
        limit = count // 10
        for i in range(1, 11):
            engine.execute(f"""
                UPDATE product SET step = {i} WHERE id IN (
                    SELECT id FROM product WHERE step IS NULL LIMIT {limit}
                )
            """)
        logging.info("--- New data found. Slicing data in 10 steps. ---")
        return "initial_training_group.train_step_1"
    else:
        engine.execute("UPDATE product SET step = 1 WHERE step IS NULL")
        logging.info("--- Sufficient new data found for single step retraining. ---")
        return "single_train_step"

def call_training_api(step, **kwargs):
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

    # PFAD B: Einzelschritt
    single_train = PythonOperator(
        task_id='single_train_step',
        python_callable=call_training_api,
        op_kwargs={'step': 1}
    )

    # PFAD A: 10 Schritte
    with TaskGroup(group_id='initial_training_group') as training_group:
        previous_step = None
        for i in range(1, 11):
            step_task = PythonOperator(
                task_id=f'train_step_{i}',
                python_callable=call_training_api,
                op_kwargs={'step': i}
            )
            if previous_step:
                previous_step >> step_task
            previous_step = step_task

    branch >> [training_group, single_train, skip_training]