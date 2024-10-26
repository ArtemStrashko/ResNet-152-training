# Not to be executed directly. Execution to be done using MLflow Project.

import sys

import config as cfg
import mlflow

from data_parallel.data_parallel import data_parallel_main


def reproduce_run(run_id):
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    # Retrieve the specified run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # Fetch parameters from the specified run
    params = run.data.params
    batch_size = int(params.get("batch_size", 32))  # Default to 32 if not found
    learning_rate = float(
        params.get("learning_rate", 1e-3)
    )  # Default to 1e-3 if not found
    epochs = int(
        params.get("epochs", cfg.MAX_N_EPOCHS)
    )  # Default to MAX_N_EPOCHS if not found

    # Start a new parent run to track the reproduced run
    with mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID) as parent_run:
        args = {
            "do_data_parallel": cfg.DO_DATA_PARALLEL,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "device": cfg.DEVICE,
            "mlflow_parent_run_id": parent_run.info.run_id,  # Pass run ID for child tracking
            "max_n_epochs": cfg.MAX_N_EPOCHS,
            "train_data_size": cfg.TRAIN_DATA_SIZE,
            "test_data_size": cfg.TEST_DATA_SIZE,
            "valid_data_size": cfg.VALID_DATA_SIZE,
        }

        # Execute the main training function
        data_parallel_main(args)

        # Log reproduced parameters for verification
        mlflow.log_params(args)

        print("Reproduced run parameters:", args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide the run ID to reproduce.")

    run_id = sys.argv[1]
    print(f"Reproducing run with run ID = {run_id}")

    reproduce_run(run_id)
