# Airflow DAG Deployment

This project ships with an example DAG for running subtitle experiments
in [Apache Airflow](https://airflow.apache.org/).

## Deploying the DAG

1. **Install dependencies**
   Ensure Airflow is installed and running. Copy any runtime dependencies from
   `requirements.txt` into your Airflow environment.
2. **Place the DAG file**
   Copy or symlink `airflow/pipeline_dag.py` into your Airflow instance's `dags/`
   directory so that it is discovered by the scheduler.
3. **Configure notifications**
   Update the `alerts@example.com` address in the DAG to the email used for
   failure notifications.
4. **Run the DAG**
   Trigger the DAG manually from the Airflow UI or CLI. Provide a JSON
   configuration matching the `SubtitleExperiment` parameters to the run to
   customize each execution.

The DAG will retry failed runs twice and send an email notification if the run
ultimately fails.
