"""
Reproduce Model Run Script

This script reproduces a model run by loading parameters from an existing
MLflow run ID and re-running the model training using those parameters.
The reproduced run is tracked in MLflow under a new parent run to distinguish
it from the original.

### Usage:
    `python reproduce.py <run_id>`

### Arguments:
    run_id (str): The ID of the original MLflow run to reproduce.

### Requirements:
- Ensure the `config.py` file includes necessary MLflow and training configurations.
- The `data_parallel_main` function should be defined in the `data_parallel` module.

This script:
    1. Retrieves parameters from the specified run ID.
    2. Starts a new parent run in MLflow for tracking.
    3. Executes the main training function with the retrieved parameters.
    4. Logs reproduced parameters for traceability.
"""

import sys

import config as cfg
import mlflow
from data_parallel import data_parallel_main


def reproduce_run(run_id):
    """
    Reproduce a model run based on parameters from an existing MLflow run.

    Args:
        run_id (str): The ID of the MLflow run whose parameters are to be reproduced.
    """

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    parent_run = mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID)
    mlflow.end_run(status="FINISHED")
    params = run.data.params
    mlflow.end_run()

    args = {
        "do_data_parallel": bool(params["do_data_parallel"]),
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "max_n_epochs": int(params["max_n_epochs"]),
        "device": cfg.DEVICE,
        "train_data_size": int(params["train_data_size"]),
        "test_data_size": int(params["test_data_size"]),
        "valid_data_size": int(params["valid_data_size"]),
        "mlflow_parent_run": parent_run,
    }
    data_parallel_main(args)


if __name__ == "__main__":
    # Check for the required argument (run ID)
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide the run ID to reproduce. Usage: python reproduce.py <run_id>"
        )

    # Parse and use the provided run ID
    run_id = sys.argv[1]
    print(f"Reproducing run with run id = {run_id}")

    # Run reproduction process
    reproduce_run(run_id)
