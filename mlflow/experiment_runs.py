import mlflow

# The tracking url is where you store your data
mlflow.set_tracking_uri(uri="http://localhost:5001")

# Here we set the experiment
mlflow.set_experiment(experiment_name="experiment_runs")

# Now setup a run
with mlflow.start_run(run_name="run1"):
    # Log a parameter to the run - something that is consistent across this run
    mlflow.log_param(key="model", value="GPT5-mini")
    # Log a metric, how well did this run go?
    mlflow.log_metric(key="accuracy", value=0.5)
    print("Run 1 done in the experiment!")

with mlflow.start_run(run_name="run2"):
    mlflow.log_param(key="model", value="GPT5")
    mlflow.log_metric(key="accuracy", value=0.9)
