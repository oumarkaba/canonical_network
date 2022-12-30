# To run shapenet experiments

# To train on so3 and test on so3
python -m canonical_network.train_shapenet --run_mode train --model pointnet --train_rotation so3 --valid_rotation so3 --batch_size 32

# To train on z and test on so3
python -m canonical_network.train_shapenet --run_mode train --model pointnet --train_rotation z --valid_rotation so3 --batch_size 32