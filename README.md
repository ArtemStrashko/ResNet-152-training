# ResNet-152 Parallel Training

This repository contains code and experiments for learning Data Parallel and Fully Sharded Data Parallel (FSDP) training of the ResNet-152 model using PyTorch. It also contains code for hyper parameter tuning with raytune and Optuna.

## Features
- Computer-vision model ResNet-152
- Data Parallel training using `torch.nn.DataParallel`
- Fully Sharded Data Parallel training using `torch.distributed.fsdp`
- Imagenette2 dataset for training and evaluation (needs to be downloaded)
- Hyperparameter tuning with `ray.tune`

## Requirements
- torch
- torchvision
- mlflow
- tqdm
- ray[tune]
- optuna

## More detailed description
`fsdp` folder contains code for [PyTorch's Fully Sharded Data Parallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html) training on multiple GPUs. FSDP divides a model across multiple GPUs by sharding parameters, gradients, and optimizer states. It fully shards these elements, reducing the memory footprint on each GPU and enabling efficient training of large models. For more details, see the [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277).

`raytune_mlflow_project` folder contains `data_parallel`, which contains code for data parallel training using `nn.DataParallel`, which keeps a copy of a model on each GPU and allows processing data in a distributed way, which speeds up training. It is much faster than FSDP, but restricted to models which can fit on a single GPU.  

`raytune_mlflow_project` folder also contains code for hyper parameter optimization with [ray.tun](https://docs.ray.io/en/latest/tune/index.html) utilizing [Optuna](https://optuna.org/) for Bayesian search. 

## Usage
Refer to the scripts for both Data Parallel and FSDP and hyper-parameter tuning implementations in `raytune_mlflow_project/data_parallel/data_parallel_train.py`, `fsdp/fsdp.py` and `raytune_mlflow_project/tune.py` respectively.
