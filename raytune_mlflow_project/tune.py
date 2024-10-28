"""
Hyperparameter Tuning Script with Ray Tune and MLflow

This script performs hyperparameter tuning for a model training function using Ray Tune, Optuna for Bayesian
optimization, and MLflow for experiment tracking. It uses data parallelism and distributed resources
for efficient tuning across multiple GPUs.

Usage:
    Execute using: `python3 /path/to/raytune_mlflow_project/tune.py`
"""

import config as cfg
import mlflow
from data_parallel import data_parallel_main
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# Set tracking URI and start parent run for MLflow logging
mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
with mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID) as parent_run:
    parent_run_id = parent_run.info.run_id  # Get parent run ID to link child runs

config_space = {
    "do_data_parallel": cfg.DO_DATA_PARALLEL,
    "batch_size": tune.choice(list(range(16, 34, 2))),
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "max_n_epochs": cfg.MAX_N_EPOCHS,
    "train_data_size": cfg.TRAIN_DATA_SIZE,
    "test_data_size": cfg.TEST_DATA_SIZE,
    "valid_data_size": cfg.VALID_DATA_SIZE,
    "device": cfg.DEVICE,
    "mlflow_parent_run_id": parent_run_id
    or None,  # Pass parent run ID to link child runs in MLflow
}


# Define ASHA Scheduler for early stopping of poor trials based on intermediate results.
scheduler = ASHAScheduler(metric="clf_accuracy", mode="max")

# Configure Optuna for Bayesian optimization to improve accuracy metric
optuna_search = OptunaSearch(
    metric="clf_accuracy",
    mode="max",
)

# Assign resources to the training function (e.g., specify GPU count)
trainable_with_resources = tune.with_resources(data_parallel_main, {"gpu": cfg.NUM_GPU})

# Set up the Ray Tune Tuner with defined hyperparameter space,
# scheduler, and search algorithm
tuner = tune.Tuner(
    trainable=trainable_with_resources,
    param_space=config_space,
    tune_config=tune.TuneConfig(
        num_samples=cfg.NUM_SAMPLES,
        search_alg=optuna_search,
        scheduler=scheduler,
    ),
)

# Execute the tuning and retrieve the best trial result
results = tuner.fit()
best_trial = results.get_best_result("clf_accuracy", "max", "last")

# Display the best trial's hyperparameters and accuracy
best_trial_config = best_trial.config
print(
    "Best trial config {batch_size, learning_rate, epochs}: ",
    [
        best_trial_config["batch_size"],
        best_trial_config["learning_rate"],
        best_trial_config["max_n_epochs"],
    ],
)
print("Best trial final clf_accuracy:", best_trial.metrics["clf_accuracy"])
