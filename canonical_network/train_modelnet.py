import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import os
from argparse import ArgumentParser
from canonical_network.prepare.modelnet_data import ModelNetDataModule
from canonical_network.models.pointcloud_classification_models import Pointnet, DGCNN, EquivariantPointcloudModel, VNPointnet
import torch


def get_hyperparams():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="equivariant_pointcloud_model",
                        help="model to train 1) equivariant_pointcloud_model 2) pointnet 3) DGCNN")
    parser.add_argument("--pred_model_type", type=str, default="DGCNN",
                        help="base encoder to use for the model 1)DGCNN 2) pointnet")
    parser.add_argument("--canon_model_type", type=str, default="vn_net",
                        help="canonicalization network type 1)vn_net")
    parser.add_argument("--pretrained", type=int, default=0,
                        help="whether the prediction network is pretrained [not implemented yet]")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--run_mode", type=str, default='train', help="different run modes 1)dryrun 2)train 3)test 4)auto_tune")

    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--patience", type=int, default=200, help="patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--decay_type', type=str, default="step", help='Decay type 1) cosine 2) step [default: cosine]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Adam or SGD [default: SGD]')

    # File handling specific arguments
    parser.add_argument("--dataset", type=str, default="modelnet", help="dataset to train on")
    parser.add_argument("--data_path", type=str, default="/network/projects/siamak_students/"
                                                         "modelnet40_normal_resampled",
                        help="path to data")
    parser.add_argument("--use_checkpointing", type=int, default=1, help="use checkpointing")
    parser.add_argument("--checkpoint_path", type=str, default="canonical_network/checkpoints", help="path to checkpoint")
    parser.add_argument("--deterministic", type=bool, default=False, help="deterministic training")
    parser.add_argument("--use_wandb", type=int, default=0, help="use wandb")
    parser.add_argument("--wandb_project", type=str, default="canonical_network", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="symmetry_group", help="wandb entity name")

    # Shapenet specific hyperparameters
    parser.add_argument("--regularization_transform", type=int, default=0, help="regularization transform")
    parser.add_argument("--normal_channel", type=bool, default=False, help="normal channel [default: False]")
    parser.add_argument("--train_rotation", type=str, default="z", help="train rotation 1)z 2)so3")
    parser.add_argument("--valid_rotation", type=str, default="so3", help="train rotation 1)z 2)so3 [default: so3 to test equivariance]")
    parser.add_argument("--augment_train_data", type=int, default=0, help="whether to scale and shift the train data [default: 0]")
    parser.add_argument("--num_classes", type=int, default=40, help="num classes of the classification problem [default: 16]")
    parser.add_argument("--num_points", type=int, default=1024, help="num of points per pointcloud [default: 2048]")
    parser.add_argument("--n_knn", type=int, default=20, help="num of nearest neighbors for DGCNN [default: 40]")
    parser.add_argument("--pooling", type=str, default="mean", help="pooling for VectorNeuron [default: mean]")

    args = parser.parse_args()
    return args
def train_pointnet():
    hyperparams = get_hyperparams()
    hyperparams.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams.wandb_project = hyperparams.wandb_project + "-" + hyperparams.dataset

    if not hyperparams.use_wandb:
        print('Wandb disable for logging.')
        os.environ["WANDB_MODE"] = "disabled"
    else:
        print('Using wandb for logging.')
        os.environ["WANDB_MODE"] = "online"

    if hyperparams.use_checkpointing:
        if hyperparams.model is not "equivariant_pointcloud_model":
            hyperparams.checkpoint_path = hyperparams.checkpoint_path + "/" + hyperparams.dataset + "/" + hyperparams.model \
                                          + "/train_rotation_" + hyperparams.train_rotation
            checkpoint_name = f"{hyperparams.model}_seed_{hyperparams.seed}"
        else:
            hyperparams.checkpoint_path = hyperparams.checkpoint_path + "/" + hyperparams.dataset + "/" + hyperparams.model \
                                          + "/" + hyperparams.pred_model_type + "/train_rotation_" + hyperparams.train_rotation
            checkpoint_name = f"{hyperparams.model}_" \
                            f"canon_model_{hyperparams.canon_model_type}"\
                            f"_seed_{hyperparams.seed}"

    wandb.init(config=hyperparams, entity=hyperparams.wandb_entity, project=hyperparams.wandb_project)
    wandb_logger = WandbLogger(project=hyperparams.wandb_project, log_model="all")

    hyperparams = wandb.config

    pl.seed_everything(hyperparams.seed)

    data = ModelNetDataModule(hyperparams)

    callbacks = []

    if hyperparams.use_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=hyperparams.checkpoint_path,
            filename= checkpoint_name,
            monitor="valid/instance_accuracy",
            mode="max"
        )
        callbacks.append(checkpoint_callback)

    early_stop_metric_callback = EarlyStopping(monitor="valid/class_accuracy", min_delta=0.0, patience=hyperparams.patience, verbose=True, mode="max")
    callbacks.append(early_stop_metric_callback)

    if hyperparams.run_mode == "test":
        model = {
            "equivariant_pointcloud_model": lambda : EquivariantPointcloudModel.load_from_checkpoint(
                checkpoint_path=hyperparams.checkpoint_path + "/" + checkpoint_name + ".ckpt",
                hyperparams=hyperparams
            ),
            "pointnet": lambda : Pointnet.load_from_checkpoint(
                checkpoint_path=hyperparams.checkpoint_path + "/" + checkpoint_name + ".ckpt",
                hyperparams=hyperparams
            ),
            "vn_pointnet": lambda : VNPointnet.load_from_checkpoint(
                checkpoint_path=hyperparams.checkpoint_path + "/" + checkpoint_name + ".ckpt",
                hyperparams=hyperparams
            ),
            "DGCNN": lambda : DGCNN.load_from_checkpoint(
                checkpoint_path=hyperparams.checkpoint_path + "/" + checkpoint_name + ".ckpt",
                hyperparams=hyperparams
            ),
        }
    else:
        model = {
            "equivariant_pointcloud_model": lambda: EquivariantPointcloudModel(hyperparams),
            "pointnet": lambda: Pointnet(hyperparams),
            "vn_pointnet": lambda: VNPointnet(hyperparams),
            "DGCNN": lambda: DGCNN(hyperparams),
        }[hyperparams.model]()

    if hyperparams.run_mode == "auto_tune":
        trainer = pl.Trainer(max_epochs=hyperparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic)
        trainer.tune(model, datamodule=data)
    elif hyperparams.run_mode == "dryrun":
        trainer = pl.Trainer(fast_dev_run=2, max_epochs=hyperparams.num_epochs, accelerator="auto", limit_train_batches=5, limit_val_batches=5, logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic)
    else:
        trainer = pl.Trainer(max_epochs=hyperparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic)

    if hyperparams.run_mode == "train":
        trainer.fit(model, datamodule=data)
    elif hyperparams.run_mode == "test":
        trainer.test(model, datamodule=data)
    else:
        raise ValueError("Invalid run mode")


def main():
    train_pointnet()

if __name__ == "__main__":
    main()
