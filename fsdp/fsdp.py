"""
fsdp_train.py

This script demonstrates fine-tuning a ResNet-152 model on the Imagenette
dataset using PyTorch's Fully Sharded Data Parallel (FSDP) for efficient
distributed training on multiple GPUs, with MLflowfor experiment tracking.

Execute with:
torchrun --nnodes 1 --nproc_per_node 4 fsdp_train.py
(when using 1 node with 4 GPUs)
"""

import functools
import os
import random
import time
from datetime import datetime

import config as cfg
import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm


class FsdpFineTune:
    """
    Basic class for fine-tuning a model using FSDP
    (Fully Sharded Data Parallel).
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        n_data_classes: int = 10,
    ):
        """Initialize FSDP training class, configure model layers,
        optimizer, and scheduler."""
        n_input_features = model.fc.in_features
        model.fc = nn.Linear(n_input_features, n_data_classes)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

    def train_epoch(self, train_loader: DataLoader, device: torch.device, epoch: int):
        "Train the model for one epoch."
        self.model.train()
        # For consistent shuffling across epochs.
        train_loader.sampler.set_epoch(epoch)

        for images, real_labels in tqdm(train_loader, leave=False):
            images, real_labels = images.to(device), real_labels.to(device)
            self.optimizer.zero_grad()
            raw_logits = self.model(images)
            loss = self.criterion(raw_logits, real_labels)
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def eval(self, eval_data_loader: DataLoader, device: torch.device):
        """Evaluate model accuracy on the provided dataset."""
        self.model.eval()
        correct = torch.tensor(0).to(device=device)
        total = torch.tensor(0).to(device=device)

        with torch.no_grad():
            for images, real_labels in tqdm(eval_data_loader, leave=False):
                images, real_labels = images.to(device), real_labels.to(device)
                raw_logits = self.model(images)
                predicted_labels = torch.argmax(raw_logits, dim=1)
                correct += (predicted_labels == real_labels).sum().item()
                total += len(real_labels)

        # Aggregate results across devices.
        dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)

        if device == 0:
            accuracy = correct.float() / total
            return accuracy.item()
        else:
            return None  # Other ranks return None

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_n_epochs: int = 100,
        patience: int = 10,
        early_stopping: bool = True,
    ):
        """Train with early stopping based on validation accuracy."""

        best_val_acc = 0.0
        epochs_to_improve = 0
        n_epochs = 0

        for epoch in range(max_n_epochs):
            n_epochs += 1
            start_time = time.time()
            self.train_epoch(train_loader=train_loader, device=device, epoch=epoch)
            train_acc = self.eval(train_loader, device)
            val_acc = self.eval(val_loader, device)
            end_time = time.time()
            time_taken = end_time - start_time

            # Log metrics only on the master process (rank 0)
            if dist.get_rank() == 0:
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

                # Early stopping
                if val_acc > best_val_acc * 1.01:
                    best_val_acc = val_acc
                    epochs_to_improve = 0
                else:
                    epochs_to_improve += 1

                early_stop = epochs_to_improve >= patience and early_stopping
            else:
                early_stop = False

            # Broadcast the early_stop signal to all ranks
            early_stop = torch.tensor(early_stop or 0).to(device)
            dist.broadcast(early_stop, src=0)
            early_stop = bool(early_stop.item())

            if early_stop:
                if dist.get_rank() == 0:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        # Return values only from the master process (rank 0)
        if dist.get_rank() == 0:
            return train_acc, val_acc, n_epochs
        else:
            return None, None, n_epochs

    def _log_metrics(
        self, epoch: int, train_acc: float, val_acc: float, time_taken: float
    ):
        """Log metrics to MLflow on the master process."""
        if dist.get_rank() == 0:
            mlflow.log_metrics(
                {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "time_taken": time_taken,
                },
                step=epoch,
            )


class ImagenetteDataLoader:
    """Class to load the Imagenette dataset with transforms
    and distributed samplers."""

    def get_data_loaders(
        self,
        rank: torch.device,
        world_size: int,
        train_data_size: int | None = None,
        valid_data_size: int | None = None,
        test_data_size: int | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Load data and create distributed data loaders."""

        train_dir = os.path.expanduser("~/imagenette2/train")
        val_dir = os.path.expanduser("~/imagenette2/val")

        # Verify data directories
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print("The data may not be downloaded. Download as follows:")
            print("wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz")
            print("tar -xvzf imagenette2.tgz")
            raise FileNotFoundError(f"{train_dir} or {val_dir} does not exist")

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

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_test_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

        val_size = int(2 * len(val_test_dataset) / 3)
        test_size = len(val_test_dataset) - val_size
        val_dataset, test_dataset = random_split(
            val_test_dataset, [val_size, test_size]
        )

        # Subsample the data if needed.
        train_dataset = self._subsample(train_dataset, train_data_size)
        val_dataset = self._subsample(val_dataset, valid_data_size)
        test_dataset = self._subsample(test_dataset, test_data_size)

        # Create samplers.
        sampler_train = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True
        )
        sampler_valid = DistributedSampler(
            val_dataset, rank=rank, num_replicas=world_size, shuffle=False
        )
        sampler_test = DistributedSampler(
            test_dataset, rank=rank, num_replicas=world_size, shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler_train,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler_valid,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler_test,
        )

        return train_loader, val_loader, test_loader

    def _subsample(self, dataset: Subset, n_samples: int | None = None):
        if n_samples is not None:
            random_indices = random.sample(range(len(dataset)), n_samples)
            return Subset(dataset, random_indices)
        return dataset


def set_random_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    """
    Initializes the process group using the NCCL backend, which allows distributed
    processes running on different GPUs to communicate efficiently during the training
    process. Here nccl statnds for NVIDIA Collective Communications Library.

    This function should be called once at the beginning of the program.
    """
    dist.init_process_group("nccl")


def cleanup_distributed():
    """
    Cleans up the process group (distributed environment) and releases the resources
    associated with it.
    """

    dist.destroy_process_group()


def fsdp_main():
    "Main function for training a model using FSDP."

    set_random_seed()
    setup_distributed()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    batch_size = cfg.PER_DEVICE_BATCH_SIZE
    max_n_epochs = cfg.MAX_N_EPOCHS
    learning_rate = cfg.LEARNING_RATE
    train_data_size = cfg.TRAIN_DATA_SIZE
    test_data_size = cfg.TEST_DATA_SIZE
    valid_data_size = cfg.VALID_DATA_SIZE

    # Log parameters to mlflow.
    if rank == 0:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_ID)
        mlflow.start_run(run_name=cfg.MLFLOW_RUN_NAME)
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_n_epochs": max_n_epochs,
                "train_data_size": cfg.TRAIN_DATA_SIZE,
                "test_data_size": cfg.TEST_DATA_SIZE,
                "valid_data_size": cfg.VALID_DATA_SIZE,
                "n_GPUs": torch.cuda.device_count(),
            }
        )

    dataset = ImagenetteDataLoader()
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        rank=rank,
        world_size=world_size,
        train_data_size=train_data_size,
        valid_data_size=valid_data_size,
        test_data_size=test_data_size,
        batch_size=batch_size,
    )

    # Set device and memory limit
    torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(cfg.MEMORY_LIMIT)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    # Load the model.
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        cpu_offload=CPUOffload(offload_params=True),
    )

    # Comments.
    # 1. The auto_wrap_policy ensures that layers with more than 100 parameters
    # are automatically sharded by FSDP. This policy is useful for large models
    # like ResNet-152, which contain many layers, allowing the framework to handle
    # only the larger layers that will benefit most from sharding.
    # 2. ShardingStrategy.FULL_SHARD means that FSDP will fully shard the model's parameters,
    # gradients, and optimizer states across all devices (GPUs). This maximizes memory savings.
    # 3. With cpu_offload=CPUOffload(offload_params=True), model parameters are moved to the CPU
    # when not in use, freeing up GPU memory for active computations. This can be beneficial in
    # environments with limited GPU memory. However, it can slow down training due to the extra
    # data transfer between CPU and GPU.

    # Initialize the trainer.
    trainer = FsdpFineTune(model=model, learning_rate=learning_rate)

    # Training loop
    start_time = datetime.now()

    train_acc, val_acc, epochs = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        device=rank,
        max_n_epochs=max_n_epochs,
    )

    end_time = datetime.now()

    test_acc = trainer.eval(test_loader, rank)

    if rank == 0:
        mlflow.log_metrics(
            {
                "final_train_acc": train_acc,
                "final_val_acc": val_acc,
                "final_test_acc": test_acc,
                "time_per_epoch": ((end_time - start_time).seconds) / epochs,
                "max_memory_per_device_GB": round(
                    torch.cuda.max_memory_allocated() / 1e9, 2
                ),
            }
        )
        mlflow.end_run()

    dist.barrier()  # Synchronize all distributed processes
    cleanup_distributed()


if __name__ == "__main__":

    # local_rank = int(os.environ["LOCAL_RANK"])
    # total_devices = int(os.environ["WORLD_SIZE"])

    # if local_rank == 0:
    #     print(f"Training on {total_devices} devices")

    # batch_size = cfg.PER_DEVICE_BATCH_SIZE * total_devices

    # if local_rank == 0:
    #     print("Per Device Batch Size = ", cfg.PER_DEVICE_BATCH_SIZE)
    #     print("Total Effective Batch Size = ", batch_size)

    # args = {
    #     "per_device_batch_size": cfg.PER_DEVICE_BATCH_SIZE,
    #     "learning_rate": cfg.LEARNING_RATE,
    #     "max_n_epochs": cfg.MAX_N_EPOCHS,
    #     "train_data_size": cfg.TRAIN_DATA_SIZE,
    #     "test_data_size": cfg.TEST_DATA_SIZE,
    #     "valid_data_size": cfg.VALID_DATA_SIZE,
    # }

    fsdp_main()
