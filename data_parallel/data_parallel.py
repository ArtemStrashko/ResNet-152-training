# Import necessary libraries and modules
import random
import torch  # Import the PyTorch library
from datetime import datetime  # Import the datetime module for time tracking
import torch.nn as nn  # Import PyTorch's neural network module
from torch.utils.data import DataLoader, random_split  # Import PyTorch data loading utilities
from torchvision import models
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm  # Import tqdm for progress tracking
import config as cfg  # Import a custom configuration module (assumed to exist)
import os  # Import the os module for operating system-related functions

# TODO: Write Distributed Data Parallel Code! Right now it is just Data Parallel.

class DataParallelFineTune:
    def __init__(self, model: nn.Module, # =models.resnet152(pretrained=True),
                #  optimizer=None,
                #  criterion=None,
                #  scheduler=None,
                 learning_rate: float = 1e-4,
                 n_data_classes: int = 10,
                 do_data_parallel: bool = False):
        self.model = model
        n_input_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(n_input_features, n_data_classes)

        if do_data_parallel and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=10, gamma=0.1)

    def train_epoch(self, train_loader: DataLoader, epoch:int):
        self.model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, leave=False):
            images, real_labels = batch
            # images, real_labels = images.to(rank), real_labels.to(rank)

            self.optimizer.zero_grad()

            raw_logits = self.model(images)

            # Mean of loss is required to be computed as calculated by each device
            loss = self.criterion(raw_logits, real_labels)
            # if do_data_parallel and torch.cuda.device_count() > 1:
            #     # Check if data parallelism is enabled and multiple GPUs are available
            #     loss = loss.mean()  # Compute the mean loss across GPUs

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        self.scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}')
        return avg_train_loss

    def eval(self, eval_data_loader: DataLoader, rank:int):
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(eval_data_loader, leave=False):
                images, real_labels = batch
                images, real_labels = images.to(rank), real_labels.to(rank)
                raw_logits = self.model(images)
                predicted_labels = torch.argmax(raw_logits, dim=1)
                correct_predictions = (predicted_labels == real_labels).sum().item()
                total_correct += correct_predictions
                total_samples += real_labels.size(0)

        accuracy = total_correct / total_samples
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

        return accuracy

class Imagenette:
    def get_data_loaders(self,
                 batch_size:int = 32,
                 num_workers:int = 4):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
            transforms.RandomHorizontalFlip(),  # Apply random horizontal flipping
            transforms.ToTensor(),              # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet stats
                                std=[0.229, 0.224, 0.225]),
            ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),             # Resize to 256 pixels on the shorter side
            transforms.CenterCrop(224),         # Center crop to 224x224
            transforms.ToTensor(),              # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet stats
                                std=[0.229, 0.224, 0.225]),
            ])

        train_dir = "~/imagenette2/train"
        val_dir = "~/imagenette2/val"

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_test_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

        val_size = int(2 * len(val_test_dataset) / 3)
        test_size = len(val_test_dataset) - val_size

        val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)

        return train_loader, val_loader, test_loader


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Main function for data parallel training
def data_parallel_main(args: dict):

    set_random_seed(1234)

    # Load the model.
    do_data_parallel = args['do_data_parallel']
    model = models.resnet152(pretrained=True)
    if do_data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.visible_devices)

    # Load the data loaders.
    batch_size = args['batch_size']
    dataset = Imagenette()
    (train_loader,
     val_loader,
     test_loader) = dataset.get_data_loaders(batch_size = batch_size)

    learning_rate = args['learning_rate']
    trainer = DataParallelFineTune(model = model, learning_rate = learning_rate)

    max_n_epochs = args["max_n_epochs"]

    # Not necessary, try without reducing data size.
    # train_data_size = args['train_data_size']  # Get the size of the training data from the arguments
    # test_data_size = args['test_data_size']  # Get the size of the test data from the arguments

    # # Training loop
    # rank = torch.device(args['device'])  # Get the device for training
    # torch.cuda.set_per_process_memory_fraction(cfg.memory_limit)  # Set GPU memory allocation limit
    # model.to(rank)  # Move the model to the specified device
    # print("Training on " + str(rank))  # Print the device being used for training
    # start_time = datetime.now()  # Record the start time for training

    # for epoch in tqdm(range(epochs)):  # Iterate through training epochs
    #     train_loss = train(model, train_loader, optimizer, epoch, rank, do_data_parallel)  # Perform training and get the average loss
    #     labels, predicted_labels, test_accuracy = test(model, test_loader, rank)  # Perform testing and get accuracy

    #     scheduler.step()  # Update the learning rate scheduler

    # end_time = datetime.now()  # Record the end time for training

    # print('Time taken per epoch (seconds): ' + str(((end_time - start_time).seconds) / epochs))  # Calculate and print the time taken per epoch

    # return {'loss': train_loss}  # Return the final training loss

if __name__ == '__main__':
    total_devices = len(cfg.visible_devices) if cfg.do_data_parallel else 1  # Determine the total number of devices used for training

    print(f"Training on {total_devices} devices")  # Print the number of devices being used for training

    batch_size = cfg.per_device_batch_size * total_devices  # Calculate the total effective batch size

    print("Per Device Batch Size = ", cfg.per_device_batch_size)  # Print the batch size per device
    print("Total Effective Batch Size = ", batch_size)  # Print the total effective batch size

    args = {'do_data_parallel': cfg.do_data_parallel,
            'batch_size': batch_size,
            'learning_rate': cfg.learning_rate,
            'epochs': cfg.epochs,
            'train_data_size': cfg.train_data_size,
            'test_data_size': cfg.test_data_size,
            'max_length': cfg.max_length,
            'model_name': cfg.model_name,
            'device': cfg.device}  # Create a dictionary of training arguments
    data_parallel_main(args)  # Start the training process
    max_memory_consumed = round(torch.cuda.max_memory_allocated() / 1e9, 2)  # Get and round the maximum GPU memory consumption
    print(f"Max Memory Consumed Per Device = {max_memory_consumed} GB")  # Print the maximum memory consumed per device
