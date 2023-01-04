# To debug shapenet experiments

# To train on so3
python -m canonical_network.train_shapenet --run_mode train --model DGCNN --train_rotation so3 --use_checkpointing 0

# To train on z
python -m canonical_network.train_shapenet --run_mode train --model pointnet --train_rotation z

# To debug modelnet experiments

# To make the dataloading faster move the data to the compute node
data_file_zipped=/network/projects/siamak_students/modelnet40_normal_resampled.zip
compute_node_data_dir=$SLURM_TMPDIR
cp $data_file_zipped $compute_node_data_dir
unzip $compute_node_data_dir/modelnet40_normal_resampled.zip -d $compute_node_data_dir
data_path=$compute_node_data_dir/modelnet40_normal_resampled

python -m canonical_network.train_modelnet --run_mode train --model DGCNN --train_rotation so3 --use_checkpointing 0 --num_workers 8 --data_path $data_path