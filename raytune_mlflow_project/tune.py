"""

Execute using 'python3 /raytune_mlflow_project/tune.py'

"""

import config as cfg
import mlflow
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from data_parallel.data_parallel import data_parallel_main

# Starts a parent run to group all the tuning runs together
# When run using MLflow Projects, this will return the run object
# created my the CLI 'mlflow run ....'
mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
parent_run = mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID)

# Important to finish the parent run, otherwise a bug in Mlflow Projects
# logs everything to parent run instead of child runs
mlflow.end_run(status="FINISHED")

config_space = {
    "do_data_parallel": cfg.DO_DATA_PARALLEL,
    "batch_size": tune.choice(list(range(16, 34, 2))),
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "epochs": cfg.MAX_N_EPOCHS,
    "device": cfg.DEVICE,
    "mlflow_parent_run": parent_run,
}

scheduler = ASHAScheduler(metric="clf_accuracy", mode="max")

# Use OptunaSearch for Bayesian optimization
optuna_search = OptunaSearch(
    metric="clf_accuracy",
    mode="max",
)

trainable_with_resources = tune.with_resources(data_parallel_main, {"gpu": cfg.NUM_GPU})

tuner = tune.Tuner(
    trainable=trainable_with_resources,
    param_space=config_space,
    tune_config=tune.TuneConfig(
        num_samples=cfg.NUM_SAMPLES,
        search_alg=optuna_search,
        scheduler=scheduler,
    ),
)

results = tuner.fit()

best_trial = results.get_best_result("clf_accuracy", "max", "last")

best_trial_config = best_trial.config
print(
    "Best trial config {batch_size, learning_rate, epochs}: ",
    [
        best_trial_config["batch_size"],
        best_trial_config["learning_rate"],
        best_trial_config["epochs"],
    ],
)

print("Best trial final clf_accuracy:", best_trial.metrics["clf_accuracy"])
