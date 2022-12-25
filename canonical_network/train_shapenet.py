import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import os

from canonical_network.prepare.shapenet_data import ShapenetPartDataModule
from canonical_network.models.pointcloud_model import SHAPENET_HYPERPARAMS, PointcloudModel
from canonical_network.models.pointcloud_base_models import Pointnet, VNPointnet

HYPERPARAMS = {"model": "pointcloud_model", "canon_model_type": "vn_pointnet", "pred_model_type": "DGCNN", "batch_size": 4, "dryrun": True, "use_wandb": False, "checkpoint": False, "num_epochs": 500, "num_workers":0, "auto_tune":False, "seed": 0, "num_parts": 50, "num_classes": 16}

def train_pointnet():
    hyperparams = HYPERPARAMS | SHAPENET_HYPERPARAMS
    wandb.login()
    wandb.init(config=hyperparams, entity="symmetry_group", project="canonical_network-shapenet")
    wandb_logger = WandbLogger(project="canonical_network-shapenet")

    hyperparams = wandb.config
    shapenet_hypeyparams = hyperparams

    if not hyperparams.use_wandb:
        print('Wandb disable for logging.')
        os.environ["WANDB_MODE"] = "disabled"
    else:
        print('Using wandb for logging.')
        os.environ["WANDB_MODE"] = "online"

    pl.seed_everything(shapenet_hypeyparams.seed)

    shapenet_data = ShapenetPartDataModule(shapenet_hypeyparams)

    checkpoint_callback = ModelCheckpoint(dirpath="canonical_network/results/shapenet/model_saves", filename= shapenet_hypeyparams.model + "_" + wandb.run.name + "_{epoch}_{valid/mean_ious:.3f}", monitor="valid/mean_ious", mode="max")
    early_stop_metric_callback = EarlyStopping(monitor="valid/mean_ious", min_delta=0.0, patience=600, verbose=True, mode="max")
    early_stop_lr_callback = EarlyStopping(monitor="lr", min_delta=0.0, patience=10000, verbose=True, mode="min", stopping_threshold=1.1e-6)
    callbacks = [checkpoint_callback, early_stop_lr_callback, early_stop_metric_callback] if shapenet_hypeyparams.checkpoint else [early_stop_lr_callback, early_stop_metric_callback]

    model = {"pointcloud_model": lambda: PointcloudModel(shapenet_hypeyparams), "pointnet": lambda: Pointnet(shapenet_hypeyparams), "vn_pointnet": lambda: VNPointnet(shapenet_hypeyparams)}[shapenet_hypeyparams.model]()

    if shapenet_hypeyparams.auto_tune:
        trainer = pl.Trainer(fast_dev_run=shapenet_hypeyparams.dryrun, max_epochs=shapenet_hypeyparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks, deterministic=False)
        trainer.tune(model, datamodule=shapenet_data)
    elif shapenet_hypeyparams.dryrun:
        trainer = pl.Trainer(fast_dev_run=False, max_epochs=2, accelerator="auto", limit_train_batches=10, limit_val_batches=10, logger=wandb_logger, callbacks=callbacks, deterministic=False)
    else:
        trainer = pl.Trainer(fast_dev_run=shapenet_hypeyparams.dryrun, max_epochs=shapenet_hypeyparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks, deterministic=False)

    trainer.fit(model, datamodule=shapenet_data)


def main():
    train_pointnet()

if __name__ == "__main__":
    main()
