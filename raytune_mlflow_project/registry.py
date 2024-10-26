"""
Execute using: python3 /home/ubuntu/model-training-coding-assignment/raytune_mlflow_project/registry.py --run_id <run_id> --model_artifact_name <model_artifact_name> --model_name <registered_model_name>
"""

import argparse

import config as cfg
import mlflow
from mlflow.tracking import MlflowClient


def register_model_by_run_id(run_id, model_artifact_name, model_name):
    """Registers a model with the MLflow Model Registry using a run ID.

    Args:
        run_id: The ID of the MLflow experiment run that contains the model to register.
        model_artifact_name: name of the model artifact as logged in the experiment.
        model_name: The name of the model to create and register in the registry.
    """
    client = MlflowClient()

    try:
        # Create a registered model if it doesn't exist
        client.create_registered_model(model_name)
        print(f"Registered model '{model_name}' created.")
    except mlflow.exceptions.RestException:
        print(f"Registered model '{model_name}' already exists.")

    # Define the model URI and create a new version
    model_uri = f"runs:/{run_id}/{model_artifact_name}"
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    artifact_uri = client.get_model_version_download_uri(model_name, mv.version)
    print(f"Model version created with Download URI: {artifact_uri}")
    return mv.version


def transition_model_to_production(model_name, version):
    """Transitions a model version to the production stage.

    Args:
        model_name: The name of the model to transition.
        version: The version of the model to transition.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage="production"
    )
    print(f"Model '{model_name}' version {version} transitioned to production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="The ID of the MLflow experiment run that contains the model to register.",
    )
    parser.add_argument(
        "--model_artifact_name",
        type=str,
        required=True,
        help="Name of the model artifact as logged in the experiment.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the model to register.",
    )

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    args = parser.parse_args()

    # Register the model
    version = register_model_by_run_id(
        args.run_id, args.model_artifact_name, args.model_name
    )

    # Transition the model to production
    transition_model_to_production(args.model_name, version)

    print("Model registered and transitioned to production successfully!")
