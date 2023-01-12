#!/usr/bin/env bash
#SBATCH --array=0-24
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --output=sbatch_out/exp_images_opt_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_images_opt_sweep.%A.%a.err
#SBATCH --job-name=exp_images_opt_sweep

module load python/3.7
module load cudatoolkit

source ~/venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

itr_arr=(5 10 20 50 100)
lr_arr=(0.1 0.05 0.01 0.005 0.001)

len=${#itr_arr[@]}
itridx=$((SLURM_ARRAY_TASK_ID%len))
lridx=$((SLURM_ARRAY_TASK_ID/len))

echo "The optimization iteration is: - ${itr_arr[$itridx]} and learning rate is - ${lr_arr[$lridx]}."

python -m canonical_network.train_images --run_mode train --model equivariant_optimization --data_mode mixed --batch_size 128 --num_optimization_iters ${itr_arr[$itridx]} --rot_opt_lr ${lr_arr[$lridx]}