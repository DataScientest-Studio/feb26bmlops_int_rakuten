# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime
# import requests

# def run_incremental_step(step_number, **kwargs):
#     # airflow gets model type from **kwargs
#     # default 'bert'
#     model_type = kwargs['dag_run'].conf.get('model_type', 'bert')
    
#     # Mapping der Endpunkte
#     endpoints = {
#         "bert": "http://localhost:8000/train/text",
#         "svm": "http://localhost:8000/train/text/linear-svm"
#     }
    
#     target_url = endpoints.get(model_type)
    
#     payload = {
#         "step": step_number, 
#         "batch_size": 1 if model_type == 'bert' else 100 
#     }
    
#     print(f"Starting training for {model_type} at {target_url} (Step {step_number})")
    
#     response = requests.post(target_url, json=payload)
    
#     if response.status_code != 200:
#         raise Exception(f"Training failed for {model_type} at step {step_number}: {response.text}")
    
#     print(f"Step {step_number} for {model_type} completed!")

# with DAG(
#     'rakuten_flexible_training',
#     start_date=datetime(2023, 1, 1),
#     schedule_interval=None,
#     catchup=False
# ) as dag:

#     prev_task = None
#     for i in range(1, 11):
#         step_task = PythonOperator(
#             task_id=f'train_step_{i}',
#             python_callable=run_incremental_step,
#             op_kwargs={'step_number': i},
#             provide_context=True # to access 'dag_run.conf'
#         )
        
#         if prev_task:
#             prev_task >> step_task
#         prev_task = step_task