import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import os
import torch

from canonical_network.prepare.nbody_data import NBodyDataModule
from canonical_network.models.euclideangraph_model import NBODY_HYPERPARAMS, EuclideanGraphModel
from canonical_network.models.euclideangraph_base_models import EGNN_vel, GNN, VNDeepSets, Transformer

# Change model here
HYPERPARAMS = {"model": "Transformer", 
               "canon_model_type": "vndeepsets", 
               "pred_model_type": "GNN", 
               "batch_size": 100, 
               "dryrun": False, 
               "use_wandb": False,
               "checkpoint": False, 
               "num_epochs": 10000, 
               "num_workers":12, 
               "auto_tune":False, 
               "seed": 0}


def train_nbody():
    hyperparams = HYPERPARAMS | NBODY_HYPERPARAMS # merges the dictionaries

    if not hyperparams["use_wandb"]:
        print('Wandb disable for logging.')
        os.environ["WANDB_MODE"] = "disabled"
    else:
        print('Using wandb for logging.')
        os.environ["WANDB_MODE"] = "online"
        
    wandb.login()
    wandb.init(config=hyperparams, entity="symmetry_group", project="canonical_network-nbody-transformer")
    wandb_logger = WandbLogger(project="canonical_network-nbody-transformer")

    hyperparams = wandb.config
    # This is passed to the model as the hyperparameters
    # Can now access data using . operator. 
    # eg. hyperparams.hidden_dim 
    nbody_hypeyparams = hyperparams

    pl.seed_everything(nbody_hypeyparams.seed)
    nbody_data = NBodyDataModule(nbody_hypeyparams)

    checkpoint_callback = ModelCheckpoint(dirpath="canonical_network/results/nbody/model_saves", filename= nbody_hypeyparams.model + "_" + wandb.run.name + "_{epoch}_{valid/loss:.3f}", monitor="valid/loss", mode="min")
    early_stop_metric_callback = EarlyStopping(monitor="valid/loss", min_delta=0.0, patience=600, verbose=True, mode="min")
    early_stop_lr_callback = EarlyStopping(monitor="lr", min_delta=0.0, patience=10000, verbose=True, mode="min", stopping_threshold=1.1e-6)
    callbacks = [checkpoint_callback, early_stop_lr_callback, early_stop_metric_callback] if nbody_hypeyparams.checkpoint else [early_stop_lr_callback, early_stop_metric_callback]

    # Instantiates model using hyperparams
    model = {"euclideangraph_model": lambda: EuclideanGraphModel(nbody_hypeyparams), 
             "EGNN": lambda: EGNN_vel(nbody_hypeyparams), 
             "GNN": lambda: GNN(nbody_hypeyparams), 
             "vndeepsets": lambda: VNDeepSets(nbody_hypeyparams),
             "Transformer": lambda: Transformer(nbody_hypeyparams),
             }[nbody_hypeyparams.model]()

    if nbody_hypeyparams.auto_tune:
        trainer = pl.Trainer(fast_dev_run=nbody_hypeyparams.dryrun, max_epochs=nbody_hypeyparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks, deterministic=False, log_every_n_steps=30)
        trainer.tune(model, datamodule=nbody_data, enable_checkpointing=nbody_hypeyparams.checkpoint)
    elif nbody_hypeyparams.dryrun:
        trainer = pl.Trainer(fast_dev_run=False, max_epochs=2, accelerator="auto", limit_train_batches=10, limit_val_batches=10, logger=wandb_logger, callbacks=callbacks, deterministic=False, enable_checkpointing=nbody_hypeyparams.checkpoint, log_every_n_steps=30)
    else:
        trainer = pl.Trainer(fast_dev_run=nbody_hypeyparams.dryrun, max_epochs=nbody_hypeyparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks, deterministic=False, enable_checkpointing=nbody_hypeyparams.checkpoint, log_every_n_steps=30)
    trainer.fit(model, datamodule=nbody_data)


def main():
    train_nbody()

if __name__ == "__main__":
    main()
