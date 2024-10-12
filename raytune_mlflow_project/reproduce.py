# Not to be executed directly. Execution to be done using MLflow Project.

import sys

from data_parallel.data_parallel import data_parallel_main
import mlflow
import config as cfg

def reproduce_run(run_id):
    
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    run = mlflow.tracking.MlflowClient().get_run(run_id)

    parent_run = mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID)

    # Important to finish the parent run, otherwise a bug in Mlflow Projects logs everything to parent run instead
    # of child runs
    mlflow.end_run(status="FINISHED")

    params = run.data.params

    mlflow.end_run()

    args = {'do_data_parallel': cfg.do_data_parallel, 'batch_size': int(params['batch_size']), 'learning_rate': float(params['learning_rate']),
            'epochs': int(params['epochs']), 'model_name': cfg.model_name, 'device': cfg.device, 'mlflow_parent_run': parent_run}
    data_parallel_main(args)

    print(parameters)


if __name__ == '__main__':

    run_id = sys.argv[1]
    print(f"Reproducing run with run id = {run_id}")

    reproduce_run(run_id)