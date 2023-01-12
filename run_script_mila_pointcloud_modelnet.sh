#!/usr/bin/env bash
#SBATCH --array=0-3
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --mem=32GB
#SBATCH --time=16:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --output=sbatch_out/exp_modelnet_model_sweep_with_simplecanon.%A.%a.out
#SBATCH --error=sbatch_err/exp_modelnet_model_sweep_with_simplecanon.%A.%a.err
#SBATCH --job-name=exp_modelnet_model_sweep_with_simplecanon

module load python/3.7
module load cudatoolkit

source ~/venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo 'Copying Dataset ...'
data_file_zipped=/network/projects/siamak_students/modelnet40_normal_resampled.zip
compute_node_data_dir=$SLURM_TMPDIR
cp $data_file_zipped $compute_node_data_dir
unzip $compute_node_data_dir/modelnet40_normal_resampled.zip -d $compute_node_data_dir
data_path=$compute_node_data_dir/modelnet40_normal_resampled
echo 'Dataset Copied ..'

#model_arr=(equivariant_pointcloud_model DGCNN pointnet)
model_arr=(DGCNN pointnet)
aug_arr=(z so3)

lenmodel=${#model_arr[@]}
modelidx=$((SLURM_ARRAY_TASK_ID%lenmodel))
augidx=$((SLURM_ARRAY_TASK_ID/lenmodel))

echo "The training configuration is: Model - ${model_arr[$modelidx]} and Train rotatiom - ${aug_arr[$augidx]}."
echo "This one uses DGCNN for pred model."

python -m canonical_network.train_modelnet \
                --run_mode train \
                --data_path $data_path \
                --model equivariant_pointcloud_model \
                --pred_model_type ${model_arr[$modelidx]} \
                --train_rotation ${aug_arr[$augidx]} \
                --batch_size 32 \
                --num_epochs 250 \
                --num_workers 8