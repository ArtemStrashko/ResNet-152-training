import functools
import os
import random
import time
from datetime import datetime

import config as cfg  # Import a custom configuration module (assumed to exist)
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
    def __init__(
        self,
        model: nn.Module,  # models.resnet152(pretrained=True),
        #  optimizer=None,
        #  criterion=None,
        #  scheduler=None,
        learning_rate: float = 1e-4,
        n_data_classes: int = 10,
    ):
        n_input_features = model.fc.in_features
        model.fc = torch.nn.Linear(n_input_features, n_data_classes)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

    def train_epoch(self, train_loader: DataLoader, rank: torch.device):
        self.model.train()

        for batch in tqdm(train_loader, leave=False):
            images, real_labels = batch
            images, real_labels = images.to(rank), real_labels.to(rank)

            self.optimizer.zero_grad()

            raw_logits = self.model(images)
            loss = self.criterion(raw_logits, real_labels)

            loss.backward()
            self.optimizer.step()
            # train_loss[0] += loss.item()
            # train_loss[1] += len(batch)

        self.scheduler.step()
        # dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

    def eval(self, eval_data_loader: DataLoader, rank: torch.device):
        self.model.eval()
        accuracy = torch.zeros(1).to(rank)

        with torch.no_grad():
            for batch in tqdm(eval_data_loader, leave=False):
                images, real_labels = batch
                images, real_labels = images.to(rank), real_labels.to(rank)
                raw_logits = self.model(images)
                predicted_labels = torch.argmax(raw_logits, dim=1)
                correct_predictions = (predicted_labels == real_labels).sum().item()
                accuracy[0] += correct_predictions

        dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
        return accuracy.item() / len(eval_data_loader.dataset)
        # if rank == 0:
        #     accuracy = accuracy.item() / len(eval_data_loader.dataset)
        #     return accuracy

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        rank: torch.device,
        max_n_epochs: int = 100,
        patience: int = 10,
        early_stopping: bool = True,
    ):

        best_val_acc = 0.0
        epochs_to_improve = 0
        n_epochs = 0

        for epoch in range(max_n_epochs):
            n_epochs += 1
            start_time = time.time()
            self.train_epoch(train_loader, rank)
            train_accuracy = self.eval(train_loader, rank)
            val_accuracy = self.eval(val_loader, rank)
            end_time = time.time()
            time_taken = end_time - start_time
            if rank == 0:
                print(
                    f"Epoch {epoch + 1}: train_acc = {train_accuracy * 100:.2f}%, val_acc = {val_accuracy * 100:.2f}%, time = {time_taken:.1f} sec."
                )
            self._log_metrics(
                epoch=epoch,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                time_taken=time_taken,
            )

            if val_accuracy > best_val_acc * 1.01:
                best_val_acc = val_accuracy
                epochs_to_improve = 0
            else:
                epochs_to_improve += 1

            if epochs_to_improve >= patience and early_stopping:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        return train_accuracy, val_accuracy, n_epochs

    def _log_metrics(
        self, epoch: int, train_accuracy: float, val_accuracy: float, time_taken: float
    ):
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("time_taken", time_taken, step=epoch)


class Imagenette:
    "Imagenette dataset"

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
        "Initialize the datasets."

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

        train_dir = os.path.expanduser("~/imagenette2/train")
        val_dir = os.path.expanduser("~/imagenette2/val")
        # check if these directories exist
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            print("The data may not be downloaded. Download as follows:")
            print("wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz")
            print("tar -xvzf imagenette2.tgz")
            raise FileNotFoundError(f"{train_dir} or {val_dir} does not exist")

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


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def fsdp_main(args_dict: dict):

    set_random_seed(1234)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    batch_size = args_dict["per_device_batch_size"]
    max_n_epochs = args_dict["max_n_epochs"]
    learning_rate = args_dict["learning_rate"]
    train_data_size = args_dict["train_data_size"]
    test_data_size = args_dict["test_data_size"]
    valid_data_size = args_dict["valid_data_size"]

    # Log parameters to mlflow.
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("max_n_epochs", max_n_epochs)
    mlflow.log_param("train_data_size", train_data_size)
    mlflow.log_param("test_data_size", test_data_size)
    mlflow.log_param("valid_data_size", valid_data_size)

    dataset = Imagenette()
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        rank=rank,
        world_size=world_size,
        train_data_size=train_data_size,
        valid_data_size=valid_data_size,
        test_data_size=test_data_size,
        batch_size=batch_size,
    )

    setup()

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    # Load the model.
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(cfg.MEMORY_LIMIT)

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        cpu_offload=CPUOffload(offload_params=True),
    )

    mlflow.log_param("n_GPUs", torch.cuda.device_count())

    # Initialize the trainer.
    trainer = FsdpFineTune(model=model, learning_rate=learning_rate)

    # Training loop
    torch.cuda.set_per_process_memory_fraction(cfg.MEMORY_LIMIT)
    start_time = datetime.now()  # Record the start time for training

    train_acc, val_acc, epochs = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        rank=rank,
        max_n_epochs=max_n_epochs,
    )

    end_time = datetime.now()

    test_acc = trainer.eval(test_loader, rank)

    mlflow.log_metric("final_train_acc", train_acc)
    mlflow.log_metric("final_val_acc", val_acc)
    mlflow.log_metric("final_test_acc", test_acc)
    mlflow.log_metric("time_per_epoch", ((end_time - start_time).seconds) / epochs)

    dist.barrier()
    cleanup()


if __name__ == "__main__":

    local_rank = int(os.environ["LOCAL_RANK"])
    total_devices = int(os.environ["WORLD_SIZE"])

    if local_rank == 0:
        print(f"Training on {total_devices} devices")

    batch_size = cfg.PER_DEVICE_BATCH_SIZE * total_devices

    if local_rank == 0:
        print("Per Device Batch Size = ", cfg.PER_DEVICE_BATCH_SIZE)
        print("Total Effective Batch Size = ", batch_size)

    args = {
        "per_device_batch_size": cfg.PER_DEVICE_BATCH_SIZE,
        "learning_rate": cfg.LEARNING_RATE,
        "max_n_epochs": cfg.MAX_N_EPOCHS,
        "train_data_size": cfg.TRAIN_DATA_SIZE,
        "test_data_size": cfg.TEST_DATA_SIZE,
        "valid_data_size": cfg.VALID_DATA_SIZE,
    }
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_ID)
    with mlflow.start_run(run_name=cfg.MLFLOW_RUN_NAME):
        mlflow.log_params(args)
        fsdp_main(args)
        max_memory_consumed = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if local_rank == 0:
            mlflow.log_metric("max_memory_per_device_GB", max_memory_consumed)
