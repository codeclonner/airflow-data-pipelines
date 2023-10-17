from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import os

with DAG(
    dag_id=os.path.basename(__file__).replace(".py", ""),
    default_args={
        'retries': 1
    },
    description="A simple tutorial DAG",
    start_date=datetime(2023, 10, 1),
    schedule_interval=timedelta(seconds=60),
    catchup=False,
    ) as dag:
        t1 = BashOperator(task_id='print_date', bash_command='date')
        t2 = BashOperator(task_id='print_text', bash_command='echo "Hello World"', depends_on_past=False, retries=3)
        
        def custom_func(**kwargs):
            print("**************************** Python ******************************")
        
        t3 = PythonOperator(task_id='print_python', python_callable=custom_func)
        
        t1 >> t2 >> t3
