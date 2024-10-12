"""

Execute using: python3 /home/ubuntu/model-training-coding-assignment/raytune_mlflow_project/registry.py --run_id <run_id> --model_artifact_name <model_artifact_name> --model_name <registered_model_name>

"""

import argparse
import mlflow
import config as cfg
from mlflow.client import MlflowClient

def register_model_by_run_id(run_id, model_artifact_name, model_name, create_registered_model):
    """Registers a model with the MLflow Model Registry using a run ID.

    Args:
        run_id: The ID of the MLflow experiment run that contains the model to register.
        model_artifact_name: name of the model artifact as logged in the experiment
        model_name: The name of the model to create and register in the registry.
    """

    client = MlflowClient()

    try:
        # create a registered model only if it doesn't exist. Will through error if it exists already
        client.create_registered_model(model_name)
    except mlflow.exceptions.RestException:
        pass

    # Create a new version of the rfr model under the registered model name
    model_uri = f"mlflow-artifacts:/{cfg.MLFLOW_EXPERIMENT_ID}/{run_id}/artifacts/{model_artifact_name}"
    mv = client.create_model_version(model_name, model_uri, run_id)
    artifact_uri = client.get_model_version_download_uri(model_name, mv.version)
    print(f"Download URI: {artifact_uri}")
    return mv.version

def transition_model_to_production(model_name, version):
    """Transitions a model version to the production stage.

    Args:
        model_name: The name of the model to transition.
        version: The version of the model to transition.
    """

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(model_name, version, "production")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="The ID of the MLflow experiment run that contains the model to register.")
    parser.add_argument("--model_artifact_name", type=str, required=True, help="Name of the model artifact as logged in the experiment")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to register.")
    
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    args = parser.parse_args()

    run_id = args.run_id
    model_name = args.model_name
    model_artifact_name = args.model_artifact_name
    create_registered_model = args.create_registered_model

    # Register the model.
    version = register_model_by_run_id(run_id, model_artifact_name, model_name, create_registered_model)

    # Transition the model to production.
    transition_model_to_production(model_name, version)

    print("Model registered and transitioned to production successfully!")