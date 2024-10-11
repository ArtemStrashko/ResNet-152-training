"""

Execute using 'python3 /home/ubuntu/model-training-coding-assignment/fsdp/fsdp.py' 

"""

import torch
from datetime import datetime
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import config as cfg
import os
import torchvision.transforms as transforms
from torch.optim import AdamW, lr_scheduler

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload
)

from torch.utils.data.distributed import DistributedSampler

import os
import functools

def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = torch.zeros(2).to(rank)

    if sampler:
        sampler.set_epoch(epoch)

    for batch in tqdm(train_loader, leave=False):
        inputs, labels = batch
        inputs, labels = inputs.to(rank), labels.to(rank)

        optimizer.zero_grad()

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss[0] += loss.item()
        total_loss[1] += len(batch)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss[0] / total_loss[1]))


def test(model, rank, world_size, test_loader):
    model.eval()

    total_accuracy = torch.zeros(1).to(rank)

    all_true_labels_in_epoch = []
    all_predictions_in_epoch = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)

            outputs = model(inputs)

            _ , predicted_labels = torch.max(outputs, 1)
            correct_predictions = torch.sum(predicted_labels == labels.data).item()
            total_accuracy[0] += correct_predictions

            all_predictions_in_epoch.extend(predicted_labels.tolist())
            all_true_labels_in_epoch.extend(labels.tolist())

        dist.all_reduce(total_accuracy, op=dist.ReduceOp.SUM)

        if rank == 0:
            print(f'Accuracy on Test Set: {total_accuracy[0]/len(test_loader.dataset):.4f}')


def get_dataloaders(data_dir, batch_size, rank, world_size):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(640),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(640),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
    data_transforms[x]) for x in ['train', 'val']}

    train_dataset = image_datasets['train']
    test_dataset = image_datasets['val']

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)
    
    train_kwargs = {'batch_size': batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': batch_size, 'sampler': sampler2}

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader 


def fsdp_main(args):

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    batch_size = args['per_device_batch_size']
    learning_rate = args['learning_rate']
    epochs = args['epochs']

    data_dir = cfg.data_dir
    train_loader, test_loader = get_dataloaders(data_dir, batch_size, rank, world_size)   

    setup()

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    
    # Modify model architecture to support Imagenet
    if local_rank == 0:
        print("Loading Model")
    
    model = args['model_name']
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg.num_classes)

    torch.cuda.set_device(local_rank)

    torch.cuda.set_per_process_memory_fraction(cfg.memory_limit)

    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD, cpu_offload=CPUOffload(offload_params=True), device_id=torch.cuda.current_device())

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    for epoch in range(1, epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch)
        test(model, rank, world_size, test_loader)
        
        scheduler.step()
    
    dist.barrier()
    cleanup()


if __name__ == '__main__':

    local_rank = int(os.environ['LOCAL_RANK'])
    
    total_devices = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        print(f"Training on {total_devices} devices")

    batch_size = cfg.per_device_batch_size * total_devices

    if local_rank == 0:
        print("Per Device Batch Size = ", cfg.per_device_batch_size)
        print("Total Effective Batch Size = ", batch_size)

    args = {'per_device_batch_size': cfg.per_device_batch_size, 'learning_rate': cfg.learning_rate,
            'epochs': cfg.epochs, 'model_name': cfg.model_name}
    
    fsdp_main(args)
    max_memory_consumed = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    
    if local_rank == 0:
        print(f"Max Memory Consumed Per Device = {max_memory_consumed} GB")