"""
data_parallel_train.py

This script demonstrates data parallel training of a deep learning
resnet152 model on the Imagenette dataset using PyTorch, with logging
via MLflow. It includes basic early stopping, model evaluation,
and results logging.

To run the training with MLflow logging, use:
    mlflow run <path_to_project_directory> -e train
"""

import os
import random
import time
from datetime import datetime

import config as cfg  # Import a custom configuration module (assumed to exist)
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


class DataParallelFineTune:
    """A basic class for fine-tuning a model with data parallelism and logging to MLflow."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        n_data_classes: int = 10,
    ):
        """Initialize the trainer with a model and hyperparameters.

        Args:
            model: nn.Module: The model to train.
            learning_rate: float: The learning rate for the optimizer.
            n_data_classes: int: The number of classes in the dataset.

        """
        if isinstance(model, nn.DataParallel):
            n_input_features = model.module.fc.in_features
            # Change the final layer to have the correct number of output features.
            model.module.fc = torch.nn.Linear(n_input_features, n_data_classes)
            model_params = model.module.fc.parameters()  # Capture new parameters here
        else:
            n_input_features = model.fc.in_features
            # Change the final layer to have the correct number of output features.
            model.fc = torch.nn.Linear(n_input_features, n_data_classes)
            model_params = model.fc.parameters()  # Capture new parameters here

        optimizer = optim.Adam(model_params, lr=learning_rate)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

    def train_epoch(
        self,
        train_loader: DataLoader,
        device: torch.device,
    ):
        """Train the model for one epoch, updating model weights.

        Args:
            train_loader: DataLoader: The DataLoader for the training set.
            device: torch.device: The device to train on.

        Returns:
            av_train_loss: float: The average training loss for the epoch.

        """
        self.model.train()
        train_loss = 0.0
        for images, real_labels in tqdm(train_loader, leave=False):
            images, real_labels = images.to(device), real_labels.to(device)
            self.optimizer.zero_grad()
            raw_logits = self.model(images)
            loss = self.criterion(raw_logits, real_labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        self.scheduler.step()
        av_train_loss = train_loss / len(train_loader)
        return av_train_loss

    def eval(self, eval_data_loader: DataLoader, device: torch.device):
        """Evaluates the model on the validation or test set.

        Args:
            eval_data_loader: DataLoader: The DataLoader for the validation or test set.
            device: torch.device: The device to evaluate on.

        Returns:
            accuracy: float: The accuracy of the model on the validation or test set.

        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_data_loader, leave=False):
                images, real_labels = batch
                images, real_labels = images.to(device), real_labels.to(device)
                raw_logits = self.model(images)
                predicted_labels = torch.argmax(raw_logits, dim=1)
                correct_predictions = (predicted_labels == real_labels).sum().item()
                total_correct += correct_predictions
                total_samples += real_labels.size(0)

        accuracy = total_correct / total_samples
        return accuracy

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_n_epochs: int = 100,
        patience: int = 10,
        early_stopping: bool = True,
        early_stop_acc_improvement: float = 0.01,
    ):
        """Train the model, applying early stopping based on validation accuracy.

        Args:
            train_loader: DataLoader: The DataLoader for the training set.
            val_loader: DataLoader: The DataLoader for the validation set.
            device: torch.device: The device to train on.
            max_n_epochs: int: The maximum number of epochs to train for.
            patience: int: The number of epochs to wait for an improvement in validation accuracy.
            early_stopping: bool: Whether to apply early stopping.

        Returns:
            train_accuracy: float: The final training accuracy.
            val_accuracy: float: The final validation accuracy.
            n_epochs: int: The number of epochs trained for.

        """
        best_val_acc = 0.0
        no_improve_epochs = 0
        n_epochs = 0

        for epoch in range(max_n_epochs):
            n_epochs += 1
            start_time = time.time()
            _ = self.train_epoch(train_loader, device)
            train_acc = self.eval(train_loader, device)
            val_acc = self.eval(val_loader, device)
            end_time = time.time()
            time_taken = end_time - start_time
            print(
                f"Epoch {epoch + 1}: train_acc = {train_acc * 100:.2f}%, "
                f"val_acc = {val_acc * 100:.2f}%, "
                f"time = {time_taken:.1f} sec."
            )

            self._log_metrics(
                epoch=epoch,
                train_acc=train_acc,
                val_acc=val_acc,
                time_taken=time_taken,
            )

            if val_acc > best_val_acc * (1.0 + early_stop_acc_improvement):
                best_val_acc = val_acc
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience and early_stopping:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        return train_acc, val_acc, n_epochs

    def _log_metrics(
        self, epoch: int, train_acc: float, val_acc: float, time_taken: float
    ):
        """Log metrics to MLflow for each epoch."""
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("time_taken", time_taken, step=epoch)


class Imagenette:
    """Class for loading the Imagenette dataset with transformations and data splitting."""

    def get_data_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        train_data_size: int | None = None,
        valid_data_size: int | None = None,
        test_data_size: int | None = None,
    ):
        """Load the data and create data loaders.

        Args:
            batch_size: int: The batch size for the data loaders.
            num_workers: int: The number of workers for data loading.
            train_data_size: int: The number of training samples to use.
            valid_data_size: int: The number of validation samples to use.
            test_data_size: int: The number of test samples to use.

        Returns:
            train_loader: DataLoader: The DataLoader for the training set.
            val_loader: DataLoader: The DataLoader for the validation set.
            test_loader: DataLoader: The DataLoader for the test set.

        """

        # Load the data.
        train_dir = os.path.expanduser("~/imagenette2/train")
        val_dir = os.path.expanduser("~/imagenette2/val")
        self._check_data_availability(train_dir, val_dir)

        # Define the transformations.
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224
                ),  # Randomly crop and resize to 224x224
                transforms.RandomHorizontalFlip(),  # Apply random horizontal flipping
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet stats
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(256),  # Resize to 256 pixels on the shorter side
                transforms.CenterCrop(224),  # Center crop to 224x224
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet stats
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Load the data.
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_test_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

        # Split the validation and test sets.
        val_size = int(2 * len(val_test_dataset) / 3)
        test_size = len(val_test_dataset) - val_size
        val_dataset, test_dataset = random_split(
            val_test_dataset, [val_size, test_size]
        )

        # Subsample the data if needed.
        train_dataset = self._subsample(
            dataset=train_dataset, n_samples=train_data_size
        )
        val_dataset = self._subsample(dataset=val_dataset, n_samples=valid_data_size)
        test_dataset = self._subsample(dataset=test_dataset, n_samples=test_data_size)

        # Create the data loaders.
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader

    def _subsample(self, dataset: Subset, n_samples: int | None = None):
        """Randomly subsample a dataset if needed."""
        if n_samples is not None:
            random_indices = random.sample(range(len(dataset)), n_samples)
            return Subset(dataset, random_indices)
        return dataset

    def _check_data_availability(self, train_dir: str, val_dir: str):
        """Check if the data is available and print instructions if not."""
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print("The data may not be downloaded. Download as follows:")
            print("wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz")
            print("tar -xvzf imagenette2.tgz")
            raise FileNotFoundError(f"{train_dir} or {val_dir} does not exist")


def set_random_seed(seed: int = 1234):
    "Set the random seed for reproducibility."
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_parallel_main(args: dict):
    """Main function to initialize model, data loaders, and execute training
    with data parallelism."""
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    set_random_seed()

    parent_run = args["mlflow_parent_run"]
    do_data_parallel = args["do_data_parallel"]
    batch_size = args["batch_size"]
    max_n_epochs = args["max_n_epochs"]
    learning_rate = args["learning_rate"]
    train_data_size = args["train_data_size"]
    test_data_size = args["test_data_size"]
    valid_data_size = args["valid_data_size"]

    # Load the model and wrap it in DataParallel if needed and if multiple GPUs are available.
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    if do_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.VISIBLE_DEVICES)

    # Load the data loaders.
    dataset = Imagenette()
    (train_loader, val_loader, test_loader) = dataset.get_data_loaders(
        batch_size=batch_size,
        train_data_size=train_data_size,
        valid_data_size=valid_data_size,
        test_data_size=test_data_size,
    )

    # Initialize the trainer.
    trainer = DataParallelFineTune(model=model, learning_rate=learning_rate)

    # Training loop
    device = torch.device(args["device"])  # Get the device for training
    torch.cuda.set_per_process_memory_fraction(cfg.MEMORY_LIMIT)
    model.to(device)  # Move the model to the specified device
    print("Training on " + str(device))  # Print the device being used for training
    start_time = datetime.now()

    # with mlflow.start_run(experiment_id=cfg.MLFLOW_EXPERIMENT_ID, nested=True):
    #     if parent_run is not None:
    #         mlflow.set_tag("mlflow.parentRunId", parent_run.info.run_id)
    #     mlflow.log_params(args)

    #     def start_run(
    #     run_id: Optional[str] = None,
    #     experiment_id: Optional[str] = None,
    #     run_name: Optional[str] = None,
    #     nested: bool = False,
    #     parent_run_id: Optional[str] = None,
    #     tags: Optional[Dict[str, Any]] = None,
    #     description: Optional[str] = None,
    #     log_system_metrics: Optional[bool] = None,
    # )

    mlflow.set_experiment(experiment_name=cfg.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=cfg.MLFLOW_RUN_NAME):
        mlflow.log_params(args)

        train_acc, val_acc, epochs = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_n_epochs=max_n_epochs,
        )

        test_acc = trainer.eval(test_loader, device)
        end_time = datetime.now()

        mlflow.log_metric("final_train_acc", train_acc)
        mlflow.log_metric("final_val_acc", val_acc)
        mlflow.log_metric("final_test_acc", test_acc)
        mlflow.log_metric("time_per_epoch", ((end_time - start_time).seconds) / epochs)

    return {"clf_accuracy": val_acc}


if __name__ == "__main__":

    total_devices = len(cfg.VISIBLE_DEVICES) if cfg.DO_DATA_PARALLEL else 1
    batch_size = cfg.PER_DEVICE_BATCH_SIZE * total_devices
    print(f"Training on {total_devices} devices")
    print("Per Device Batch Size = ", cfg.PER_DEVICE_BATCH_SIZE)
    print("Total Effective Batch Size = ", batch_size)

    args = {
        "do_data_parallel": cfg.DO_DATA_PARALLEL,
        "batch_size": batch_size,
        "learning_rate": cfg.LEARNING_RATE,
        "max_n_epochs": cfg.MAX_N_EPOCHS,
        "train_data_size": cfg.TRAIN_DATA_SIZE,
        "test_data_size": cfg.TEST_DATA_SIZE,
        "valid_data_size": cfg.VALID_DATA_SIZE,
        "device": cfg.DEVICE,
        "mlflow_parent_run": cfg.MLFLOW_PARENT_RUN,
    }

    data_parallel_main(args)
    max_memory_consumed = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"Max memory consumed: {max_memory_consumed} GB")
