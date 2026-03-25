from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
import logging
from datetime import datetime, timedelta
import os
import pandas as pd
import sqlalchemy

RAW_DATA_PATH = "/opt/airflow/data/raw/"
#DB_CONN = "postgresql://airflow:airflow@postgres:5432/dst_db"
DB_CONN = "postgresql://postgres:postgres@db:5432/dst_db"

def ingest_csv():
    engine = sqlalchemy.create_engine(DB_CONN)
    files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]
    
    if not files:
        logging.info("--- CHECK: Now new CSV-files in folder 'raw' found. ---")
        return "No files found"

    logging.info(f"--- START: {len(files)} new file(s) found. Starting import... ---")
    for file in files:
        path = os.path.join(RAW_DATA_PATH, file)
        df = pd.read_csv(path)
        # mark new data
        df['step'] = None
        df['train'] = None
        # add to DB table
        df.to_sql('Product', engine, if_exists='append', index=False)
        # archive/move data
        os.rename(path, path + ".processed")
        print(f"Ingested {file}")
        logging.info(f"File {file} successfully loaded in Postgres.")

with DAG('rakuten_ingestion', 
         start_date=days_ago(2),
         schedule_interval='@hourly', 
         catchup=False
         ) as dag:

    wait_for_file = FileSensor(
        task_id='wait_for_csv', 
        filepath=RAW_DATA_PATH, 
        poke_interval=5,
        timeout=10,
        mode='poke',
        soft_fail=True,
        dag=dag)
    
    load_data = PythonOperator(
        task_id='load_to_db', 
        python_callable=ingest_csv)

    wait_for_file >> load_data

# alternative anstatt FileSensor -> PythonOperator
#     def check_for_files():
#     import os
#     files = [f for f in os.listdir('/opt/airflow/data/raw/') if f.endswith('.csv')]
#     if not files:
#         from airflow.exceptions import AirflowSkipException
#         raise AirflowSkipException("Keine Dateien gefunden, überspringe Rest.")
#     print(f"{len(files)} Dateien gefunden.")

# # In deinem DAG:
# check_task = PythonOperator(
#     task_id='check_files',
#     python_callable=check_for_files
#)