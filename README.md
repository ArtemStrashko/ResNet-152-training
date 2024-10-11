# ResNet-152 Parallel Training

This repository contains code and experiments for learning Data Parallel and Fully Sharded Data Parallel (FSDP) training of the ResNet-152 model using PyTorch. It also contains code for hyper parameter tuning with raytune.

## Features
- Computer-vision model ResNet-152
- Data Parallel training using `torch.nn.DataParallel`
- Fully Sharded Data Parallel training using `torch.distributed.fsdp`
- Imagenette2 dataset for training and evaluation
- Hyperparameter tuning with `ray.tune`

## Requirements
- torch>=2.0.1
- torchvision>=0.15.2
- mlflow>=2.7.0
- tqdm>=4.66.1
- ray[tune]>=2.7.0

## Usage
Refer to the scripts for both Data Parallel and FSDP and hyper-parameter tuning  implementations in `data_parallel/data_parallel.py`, `fsdp/fsdp.py` and `ray_tune/ray_tune.py` respectively.

Feel free to explore and modify the code to experiment with different training strategies!