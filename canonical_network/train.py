from asyncio.log import logger
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

import canonical_network.utils as utils
from canonical_network.prepare.digits_data import DigitsDataModule
from canonical_network.models.set_model import SET_HYPERPARAMS, SetModel
from canonical_network.models.base_models import DeepSets, Transformer

HYPERPARAMS = {"model": "transformer","batch_size": 64, "dryrun": False, "num_epochs": 100, "num_workers":0, "auto_tune":False}

def train_digits():
    hyperparams = HYPERPARAMS | SET_HYPERPARAMS
    wandb.login()
    wandb.init(config=hyperparams)
    wandb_logger = WandbLogger(project="canonical_network-digits", log_model="all")
    
    hyperparams = wandb.config
    set_hypeyparams = utils.dict_to_object(hyperparams)

    set_data = DigitsDataModule(set_hypeyparams)
    
    checkpoint_callback = ModelCheckpoint(dirpath="canonical_network/results/digits/model_saves", filename= set_hypeyparams.model + "_" + wandb.run.name + "_{epoch}_{valid/f1_score:.3f}", monitor="valid/f1_score", mode="max")
    early_stop_metric_callback = EarlyStopping(monitor="valid/f1_score", min_delta=0.0, patience=25, verbose=True, mode="max")
    early_stop_lr_callback = EarlyStopping(monitor="lr", min_delta=0.0, patience=10000, verbose=True, mode="min", stopping_threshold=1.1e-6)
    callbacks = [checkpoint_callback, early_stop_lr_callback, early_stop_metric_callback]

    model = {"set_model": lambda: SetModel(set_hypeyparams), "deepsets": lambda: DeepSets(set_hypeyparams), "transformer": lambda: Transformer(set_hypeyparams)}[set_hypeyparams.model]()

    if set_hypeyparams.auto_tune:
        trainer = pl.Trainer(fast_dev_run=set_hypeyparams.dryrun, max_epochs=set_hypeyparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks)
        trainer.tune(model, datamodule=set_data)
    elif set_hypeyparams.dryrun:
        trainer = pl.Trainer(fast_dev_run=set_hypeyparams.dryrun, max_epochs=set_hypeyparams.num_epochs, accelerator="auto", limit_train_batches=2, limit_val_batches=2, logger=wandb_logger, callbacks=callbacks)
    else:
        trainer = pl.Trainer(fast_dev_run=set_hypeyparams.dryrun, max_epochs=set_hypeyparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks)

    trainer.fit(model, datamodule=set_data)


def main():
    train_digits()

if __name__ == "__main__":
    main()
