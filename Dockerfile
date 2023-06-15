FROM apache/airflow:latest
ENV PYTHONPATH "${PYTHONPATH}:${AIRFLOW_HOME}"
COPY requirements.txt requirements.txt
RUN mkdir /opt/airflow/output
RUN pip install -r requirements.txt