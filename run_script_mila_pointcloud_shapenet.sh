#!/usr/bin/env bash
#SBATCH --array=0-3
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
##SBATCH --reservation=DGXA100
#SBATCH --mem=32GB
#SBATCH --time=28:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --output=sbatch_out/exp_shapenet_model_sweep.%A.%a.out
#SBATCH --error=sbatch_err/exp_shapenet_model_sweep.%A.%a.err
#SBATCH --job-name=exp_shapenet_model_sweep

module load python/3.7
module load cudatoolkit

source ~/venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo 'Copying Dataset ...'
data_file_zipped=/network/projects/siamak_students/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
compute_node_data_dir=$SLURM_TMPDIR
cp $data_file_zipped $compute_node_data_dir
unzip $compute_node_data_dir/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip -d $compute_node_data_dir
data_path=$compute_node_data_dir/shapenetcore_partanno_segmentation_benchmark_v0_normal
echo 'Dataset Copied ..'

model_arr=(equivariant_pointcloud_model DGCNN)
aug_arr=(z so3)

lenmodel=${#model_arr[@]}
modelidx=$((SLURM_ARRAY_TASK_ID%lenmodel))
augidx=$((SLURM_ARRAY_TASK_ID/lenmodel))

echo "The training configuration is: Model - ${model_arr[$modelidx]} and Train rotation - ${aug_arr[$augidx]}."


python -m canonical_network.train_shapenet \
                --run_mode train \
                --data_path $data_path \
                --model ${model_arr[$modelidx]} \
                --pred_model_type DGCNN \
                --train_rotation ${aug_arr[$augidx]} \
                --valid_rotation so3 \
                --batch_size 32 \
                --optimizer SGD \
                --num_epochs 250 \
                --num_workers 8