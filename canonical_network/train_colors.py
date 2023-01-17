import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import os

from canonical_network.prepare.colors_data import ColorsDataModule
from canonical_network.models.colouring_base_models import UNet

HYPERPARAMS = {"dataset":"celeba", "model_type":"canonical", "p_drop": 0.0, "use_max": False, "batch_size": 32, "dryrun": False, "use_wandb": False, "checkpoint": False, "num_epochs": 100, "num_workers":0, "auto_tune":False, "seed": 0, "learning_rate": 1e-3, "sort_reg":1e-3, "patience": 50, "parameters_factor": 2}

def train_colors():
    hyperparams = HYPERPARAMS

    if not hyperparams["use_wandb"]:
        print('Wandb disable for logging.')
        os.environ["WANDB_MODE"] = "disabled"
    else:
        print('Using wandb for logging.')
        os.environ["WANDB_MODE"] = "online"

    wandb.login()
    wandb.init(config=hyperparams, entity="symmetry_group", project="canonical_network-colors")
    wandb_logger = WandbLogger(project="canonical_network-colors")

    hyperparams = wandb.config
    color_hypeyparams = hyperparams

    pl.seed_everything(color_hypeyparams.seed)

    color_data = ColorsDataModule(color_hypeyparams)

    checkpoint_callback = ModelCheckpoint(dirpath="canonical_network/results/nbody/model_saves", filename= color_hypeyparams.model_type + "_" + wandb.run.name + "_{epoch}_{valid/loss:.3f}", monitor="valid/loss", mode="min")
    early_stop_metric_callback = EarlyStopping(monitor="valid/loss", min_delta=0.0, patience=600, verbose=True, mode="min")
    early_stop_lr_callback = EarlyStopping(monitor="lr", min_delta=0.0, patience=10000, verbose=True, mode="min", stopping_threshold=1.1e-6)
    callbacks = [checkpoint_callback, early_stop_lr_callback, early_stop_metric_callback] if color_hypeyparams.checkpoint else [early_stop_lr_callback, early_stop_metric_callback]

    model = UNet(color_hypeyparams)

    if color_hypeyparams.auto_tune:
        trainer = pl.Trainer(fast_dev_run=color_hypeyparams.dryrun, max_epochs=color_hypeyparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks, deterministic=False)
        trainer.tune(model, datamodule=color_data, enable_checkpointing=color_hypeyparams.checkpoint)
    elif color_hypeyparams.dryrun:
        trainer = pl.Trainer(fast_dev_run=False, max_epochs=2, accelerator="auto", limit_train_batches=10, limit_val_batches=10, logger=wandb_logger, callbacks=callbacks, deterministic=False, enable_checkpointing=color_hypeyparams.checkpoint)
    else:
        trainer = pl.Trainer(fast_dev_run=color_hypeyparams.dryrun, max_epochs=color_hypeyparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks, deterministic=False, enable_checkpointing=color_hypeyparams.checkpoint)

    trainer.fit(model, datamodule=color_data)


def main():
    train_colors()

if __name__ == "__main__":
    main()
