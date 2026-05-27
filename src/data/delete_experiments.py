import mlflow
import dagshub

dagshub.init(repo_owner="Jozela", repo_name="IISVaje", mlflow=True)

client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments(view_type=3)

for exp in experiments:
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    for run in runs:
        client.delete_run(run.info.run_id)
        print(f"Deleted run {run.info.run_id} from {exp.name}")

print("Done")