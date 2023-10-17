from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

def print_python_version():
    import sys
    print(sys.version)
    
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 13)
}

with DAG('python_version_dag',
         default_args=default_args,
         schedule_interval='@once') as dag:
    print_python_version = PythonOperator(
        task_id='print_python_version',
        python_callable=print_python_version
    )