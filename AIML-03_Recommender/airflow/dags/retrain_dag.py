from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def retrain_model():
    print("Retraining Two-Tower model...")

default_args = {'owner': 'himanshu', 'retries': 1, 'retry_delay': timedelta(minutes=5)}

with DAG('recommender_retrain', default_args=default_args, schedule_interval='@weekly', start_date=datetime(2026, 10, 1)) as dag:
    retrain_task = PythonOperator(task_id='retrain_two_tower', python_callable=retrain_model)