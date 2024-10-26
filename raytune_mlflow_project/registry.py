"""
Model Registration and Deployment Script

This script registers a model to the MLflow Model Registry using a specified run ID and model artifact name,
creates a new model version, and transitions it to the production stage.

Usage:
    python3 /path/to/raytune_mlflow_project/registry.py --run_id <run_id> --model_artifact_name <model_artifact_name> --model_name <registered_model_name>
"""

import argparse

import config as cfg
import mlflow
from mlflow.tracking import MlflowClient


def register_model_by_run_id(run_id, model_artifact_name, model_name):
    """
    Registers a model with the MLflow Model Registry using a given run ID.

    Args:
        run_id (str): The ID of the MLflow experiment run containing the model to register.
        model_artifact_name (str): Name of the model artifact as logged in the experiment.
        model_name (str): The name of the model to create and register in the registry.

    Returns:
        int: Version number of the newly created model version.
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
    model_version = client.create_model_version(
        name=model_name, source=model_uri, run_id=run_id
    )
    artifact_uri = client.get_model_version_download_uri(
        model_name, model_version.version
    )
    print(
        f"Model version {model_version.version} created with Download URI: {artifact_uri}"
    )

    return model_version.version


def transition_model_to_production(model_name: str, version: int):
    """
    Transitions a specified model version to the production
    stage in the MLflow Model Registry.

    Args:
        model_name (str): The name of the registered model.
        version (int): The version of the model to transition to production.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )
    print(f"Model '{model_name}' version {version} transitioned to Production stage.")


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

    # Set MLflow tracking URI from config
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    # Parse arguments
    args = parser.parse_args()

    # Register the model and retrieve the created version number
    version = register_model_by_run_id(
        args.run_id, args.model_artifact_name, args.model_name
    )

    # Transition the newly registered model version to production
    transition_model_to_production(args.model_name, version)

    print("Model registered and transitioned to production successfully!")
