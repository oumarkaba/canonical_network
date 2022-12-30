import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import os
from argparse import ArgumentParser
from canonical_network.prepare.shapenet_data import ShapenetPartDataModule
from canonical_network.models.pointcloud_models import Pointnet, VNPointnet, DGCNN, EquivariantPointcloudModel
import torch

def get_hyperparams():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="equivariant_pointcloud_model",
                        help="model to train 1) equivariant_pointcloud_model 2) pointnet 3) vn_pointnet 4) DGCNN")
    parser.add_argument("--pred_model_type", type=str, default="DGCNN",
                        help="base encoder to use for the model 1)DGCNN 2) pointnet")
    parser.add_argument("--canon_model_type", type=str, default="vn_pointnet",
                        help="canonicalization network type 1)vn_pointnet")
    parser.add_argument("--pretrained", type=int, default=0,
                        help="whether the prediction network is pretrained")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--run_mode", type=str, default='dryrun', help="different run modes 1)dryrun 2)train 3)test 4)auto_tune")

    parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (for SGD it is multiplied by 100) [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
    parser.add_argument('--decay_type', type=str, default="cosine", help='Decay type 1) cosine 2) step [default: cosine]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD or SGD_built_in [default: SGD]')
    parser.add_argument("--pooling", type=str, default="mean",
                        help="pooling ")

    # File handling specific arguments
    parser.add_argument("--dataset", type=str, default="shapenet", help="dataset to train on")
    parser.add_argument("--data_path", type=str, default="canonical_network/data", help="path to data")
    parser.add_argument("--checkpoint_path", type=str, default="canonical_network/results", help="path to checkpoint")
    parser.add_argument("--deterministic", type=bool, default=False, help="deterministic training")
    parser.add_argument("--use_wandb", type=int, default=0, help="use wandb")
    parser.add_argument("--wandb_project", type=str, default="canonical_network-shapenet", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="symmetry_group", help="wandb entity name")

    # Shapenet specific hyperparameters
    parser.add_argument("--regularization_transform", type=int, default=0, help="regularization transform")
    parser.add_argument("--normal_channel", type=bool, default=False, help="normal channel")
    parser.add_argument("--train_rotation", type=str, default="z", help="train rotation 1)z 2)so3")
    parser.add_argument("--valid_rotation", type=str, default="z", help="train rotation 1)z 2)so3")
    parser.add_argument("--augment_train_data", type=int, default=1, help="whether to scale and shift the train data")
    parser.add_argument("--num_classes", type=int, default=16, help="num classes of the classification problem")
    parser.add_argument("--num_parts", type=int, default=50, help="num of parts")
    parser.add_argument("--num_points", type=int, default=2048, help="num of points")
    parser.add_argument("--n_knn", type=int, default=40, help="num of nearest neighbors")

    args = parser.parse_args()
    return args
def train_pointnet():
    hyperparams = get_hyperparams()
    hyperparams.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams.data_path = hyperparams.data_path + "/" + hyperparams.dataset
    hyperparams.checkpoint_path = hyperparams.checkpoint_path + "/" + hyperparams.dataset + "/" + hyperparams.model \
                                  + "/" + hyperparams.pred_model_type + "/train_rotation_" + hyperparams.train_rotation
    hyperparams.wandb_project = hyperparams.wandb_project + "-" + hyperparams.dataset

    if not hyperparams.use_wandb:
        print('Wandb disable for logging.')
        os.environ["WANDB_MODE"] = "disabled"
    else:
        print('Using wandb for logging.')
        os.environ["WANDB_MODE"] = "online"

    wandb.init(config=hyperparams, entity=hyperparams.wandb_entity, project=hyperparams.wandb_project)
    wandb_logger = WandbLogger(project=hyperparams.wandb_project, log_model="all")

    hyperparams = wandb.config
    shapenet_hypeyparams = hyperparams

    pl.seed_everything(shapenet_hypeyparams.seed)

    shapenet_data = ShapenetPartDataModule(shapenet_hypeyparams)

    if hyperparams.model is not "equivariant_pointcloud_model":
        checkpoint_name = f"{hyperparams.model}_seed_{hyperparams.seed}"
    else:
        checkpoint_name = f"{hyperparams.model}_" \
                        f"canon_model_{hyperparams.canon_model_type}"\
                        f"_seed_{hyperparams.seed}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint_path,
        filename= checkpoint_name,
        monitor="valid/instance_avg_iou",
        mode="max"
    )
    early_stop_metric_callback = EarlyStopping(monitor="valid/class_avg_iou", min_delta=0.0, patience=hyperparams.patience, verbose=True, mode="max")
    callbacks = [checkpoint_callback, early_stop_metric_callback]

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
            "equivariant_pointcloud_model": lambda: EquivariantPointcloudModel(shapenet_hypeyparams),
            "pointnet": lambda: Pointnet(shapenet_hypeyparams),
            "vn_pointnet": lambda: VNPointnet(shapenet_hypeyparams),
            "DGCNN": lambda: DGCNN(shapenet_hypeyparams),
        }[shapenet_hypeyparams.model]()

    if shapenet_hypeyparams.run_mode == "auto_tune":
        trainer = pl.Trainer(max_epochs=shapenet_hypeyparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic)
        trainer.tune(model, datamodule=shapenet_data)
    elif shapenet_hypeyparams.run_mode == "dryrun":
        trainer = pl.Trainer(fast_dev_run=2, max_epochs=shapenet_hypeyparams.num_epochs, accelerator="auto", limit_train_batches=5, limit_val_batches=5, logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic)
    else:
        trainer = pl.Trainer(max_epochs=shapenet_hypeyparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic)

    if hyperparams.run_mode == "train":
        trainer.fit(model, datamodule=shapenet_data)

    #trainer.test(model, datamodule=shapenet_data)


def main():
    train_pointnet()

if __name__ == "__main__":
    main()
