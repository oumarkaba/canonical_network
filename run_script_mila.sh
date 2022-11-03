#!/bin/bash

module load python/3.7
module load cudatoolkit

source ~/venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8



# To run vanilla model
python -m canonical_network.train_images --run_mode train --model vanilla --batch_size 128 --use_wandb 1

# To run equivariant model
python -m canonical_network.train_images --run_mode train --model equivariant --kernel_size 28 --num_filters 16 --group_type rotation --batch_size 128 --use_wandb 1


# TO run equivariant model in rotatedMNIST
python -m canonical_network.train_images --run_mode test --model equivariant --canonization_kernel_size 7 --canonization_num_layers 3 --canonization_out_channels 16 --group_type rotation --batch_size 128 --num_rotations 8 --save_canonized_images 1

# To run equivariant model in CIFAR10
python -m canonical_network.train_images --run_mode train --model equivariant --canonization_kernel_size 32 --canonization_num_layers 1 --canonization_out_channels 16 --group_type roto-reflection --batch_size 128 --num_rotations 4 --save_canonized_images 1 --dataset cifar10 --base_encoder resnet18 --data_path ~/scratch