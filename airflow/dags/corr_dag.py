import datetime

from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import correlation_search

args={'owner': 'airflow'}

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

corr_dag = DAG(
    dag_id = "corr_dag",
    default_args=args,
    schedule_interval='@hourly',
    dagrun_timeout=timedelta(minutes=60),
    description='Correlations Search DAG',
    start_date = datetime.datetime.now()
)

with corr_dag:
    count_corr = PythonOperator(
        task_id='py_corr_count',
        python_callable=correlation_search.main
        )


if __name__ == "__main__":
        corr_dag.cli()