"""
Reproduce Model Run Script

This script reproduces a model run by loading parameters from an existing MLflow run ID and
re-running the model training using those parameters. The reproduced run is tracked in MLflow
under a new parent run to distinguish it from the original.

Usage:
    Execute within an MLflow Project using:
    `mlflow run <project_directory> -e reproduce -P run_id=<run_id>`

Arguments:
    run_id (str): The ID of the original MLflow run to reproduce.
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

    This function:
        1. Retrieves parameters from the specified run.
        2. Starts a new parent run in MLflow for tracking.
        3. Executes the main training function with the retrieved parameters.
        4. Logs reproduced parameters for traceability.
    """
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    # Retrieve parameters from the specified run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params

    # Extract and parse parameters, using defaults if necessary
    batch_size = int(params.get("batch_size", 32))  # Default to 32 if missing
    learning_rate = float(
        params.get("learning_rate", 1e-3)
    )  # Default to 1e-3 if missing
    epochs = int(
        params.get("epochs", cfg.MAX_N_EPOCHS)
    )  # Default to config value if missing

    # Start a new MLflow run to track this reproduction process
    with mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID) as parent_run:
        args = {
            "do_data_parallel": cfg.DO_DATA_PARALLEL,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "device": cfg.DEVICE,
            "mlflow_parent_run_id": parent_run.info.run_id,  # Pass parent run ID
            "max_n_epochs": cfg.MAX_N_EPOCHS,
            "train_data_size": cfg.TRAIN_DATA_SIZE,
            "test_data_size": cfg.TEST_DATA_SIZE,
            "valid_data_size": cfg.VALID_DATA_SIZE,
        }

        # Execute the main training function with the specified parameters
        data_parallel_main(args)

        # Log parameters for this reproduced run to MLflow
        mlflow.log_params(args)

        print("Reproduced run parameters:", args)


if __name__ == "__main__":
    # Check for the required argument (run ID)
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide the run ID to reproduce. Usage: python reproduce.py <run_id>"
        )

    # Parse and use the provided run ID
    run_id = sys.argv[1]
    print(f"Reproducing run with run ID = {run_id}")

    # Run reproduction process
    reproduce_run(run_id)
