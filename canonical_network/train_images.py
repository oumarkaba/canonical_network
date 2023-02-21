import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from argparse import ArgumentParser
import os
import torch
from tqdm import tqdm

from canonical_network.prepare import RotatedMNISTDataModule, CIFAR10DataModule
from canonical_network.models.image_model import LitClassifier

def get_hyperparams():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="vanilla", help="model to train 1) vanilla 2) equivariant 3) canonized_pca 4) equivariant_optimization")
    parser.add_argument("--base_encoder", type=str, default="cnn",
                        help="base encoder to use for the model 1)cnn 2)resnet18 3)resnet50 4)resnet101 5)vit 6)rotation_eqv_cnn 7)rotoreflection_eqv_cnn")
    parser.add_argument("--pretrained", type=int, default=0,
                        help="base encoder to use for the model 1)cnn 2)resnet18 3)resnet50 4)resnet101 5)vit 6)rotation_eqv_cnn 7)rotoreflection_eqv_cnn")
    parser.add_argument("--data_mode", type=str, default='image',
                        help="different run modes 1)image 2)mixed (mixed has both image and pointcloud grid representation for deepsets)")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--run_mode", type=str, default='dryrun', help="different run modes 1)dryrun 2)train 3)test 4)auto_tune")
    parser.add_argument("--use_wandb", type=int, default=True, help="use wandb")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--dataset", type=str, default="rotated_mnist", help="dataset to train on")
    parser.add_argument("--data_path", type=str, default="canonical_network/data", help="path to data")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--canonization_kernel_size", type=int, default=3, help="kernel size for canonization layer")
    parser.add_argument("--canonization_out_channels", type=int, default=16,
                        help="number of equivariant output channels for the canonization network")
    parser.add_argument("--canonization_num_layers", type=int, default=3,
                        help="number of equivariant output channels for the canonization network")
    parser.add_argument("--canonization_beta", type=float, default=1.0,
                        help="sharpness of the canonization network output")
    parser.add_argument("--group_type", type=str, default="rotation", help="group type for equivariance 1) rotation 2) roto-reflection")
    parser.add_argument("--num_rotations", type=int, default=4, help="order of the group")
    parser.add_argument("--checkpoint_path", type=str, default="canonical_network/results", help="path to checkpoint")
    parser.add_argument("--deterministic", type=bool, default=False, help="deterministic training")
    parser.add_argument("--wandb_project", type=str, default="canonical_network", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="cyanogenoid", help="wandb entity name")
    parser.add_argument("--save_canonized_images", type=int, default=0, help="save canonized images")
    parser.add_argument("--check_invariance", type=int, default=0, help="check if the network is invariant")
    parser.add_argument("--num_channels", type=int, default=20, help="num_channels for equivariant cnn base encoder")

    # Hyperparameters for the energy based model and Deepset
    parser.add_argument("--num_layers", type=int, default=6, help="number of deepset layers")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension for deepset")
    parser.add_argument("--layer_pooling", type=str, default="max", help="pooling for deepset intermediate layers")
    parser.add_argument("--final_pooling", type=str, default="max", help="pooling for deepset final layer")
    parser.add_argument("--num_optimization_iters", type=int, default=20, help="number of optimization iterations for the energy based model")
    parser.add_argument("--rot_opt_lr", type=float, default=0.01, help="number of samples for the energy based model")
    parser.add_argument("--implicit", type=int, default=0, help="whether to use implicit rotation optimization")
    args = parser.parse_args()
    return args


def train_images():
    hyperparams = get_hyperparams()
    hyperparams.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams.data_path = hyperparams.data_path + "/" + hyperparams.dataset
    hyperparams.checkpoint_path = hyperparams.checkpoint_path + "/" + hyperparams.dataset + "/" + hyperparams.model \
                                  + "/" + hyperparams.base_encoder
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

    pl.seed_everything(hyperparams.seed)

    if hyperparams.dataset == "rotated_mnist":
        image_data = RotatedMNISTDataModule(hyperparams, mode=hyperparams.data_mode)
    elif hyperparams.dataset == "cifar10":
        image_data = CIFAR10DataModule(hyperparams)
    else:
        raise NotImplementedError("Dataset not implemented")

    if hyperparams.model == "vanilla":
        checkpoint_name = f"{hyperparams.model}_seed_{hyperparams.seed}"
    else:
        checkpoint_name = f"{hyperparams.model}_kernel_{hyperparams.canonization_kernel_size}_" \
                          f"num-layer_{hyperparams.canonization_num_layers}_{hyperparams.group_type}_" \
                          f"{hyperparams.num_rotations}_seed_{hyperparams.seed}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=hyperparams.checkpoint_path,
        filename= checkpoint_name,
        monitor="val/acc",
        mode="max"
    )
    early_stop_metric_callback = EarlyStopping(monitor="val/acc", min_delta=0.0, patience=hyperparams.patience, verbose=True, mode="max")
    callbacks = [checkpoint_callback, early_stop_metric_callback]

    if hyperparams.run_mode == "test":
        model = LitClassifier.load_from_checkpoint(
            checkpoint_path=hyperparams.checkpoint_path + "/" + checkpoint_name + ".ckpt",
            hyperparams=hyperparams
        )
        model.freeze()
        model.eval()
    else:
        model = LitClassifier(hyperparams)

    if hyperparams.model == "equivariant":
        wandb.watch(model.network.canonization_network, log='all')

    if hyperparams.run_mode == "auto_tune":
        trainer = pl.Trainer(max_epochs=hyperparams.num_epochs, accelerator="auto", auto_scale_batch_size=True, auto_lr_find=True, logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic, inference_mode=False)
        trainer.tune(model, datamodule=image_data)
    elif hyperparams.run_mode == "dryrun":
        trainer = pl.Trainer(fast_dev_run=2, max_epochs=hyperparams.num_epochs, accelerator="auto", limit_train_batches=5, limit_val_batches=5, logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic, inference_mode=False)
    else:
        trainer = pl.Trainer(max_epochs=hyperparams.num_epochs, accelerator="auto", logger=wandb_logger, callbacks=callbacks, deterministic=hyperparams.deterministic, inference_mode=False, limit_val_batches=100, check_val_every_n_epoch=10)

    if hyperparams.run_mode == "train":
        trainer.fit(model, datamodule=image_data)

    trainer.test(model, datamodule=image_data)

    # This is just for sanity check and verify that the metric logging is working fine.
    #custom_test(model, image_data)



def main():
    train_images()

def custom_test(model, datamodule):
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    model.eval()
    with torch.no_grad():
        preds_list = []
        y_list = []
        for batch in tqdm(test_loader):
            x, y = batch
            preds_list.append(model(x))
            y_list.append(y)
        preds = torch.cat(preds_list, dim=0)
        y = torch.cat(y_list, dim=0)
        acc = (preds == y).float().mean()
        print("Test accuracy: ", acc)
        acc_per_class = []
        for i in range(model.num_classes):
            acc_per_class.append((preds[y == i] == y[y == i]).float().mean())
        print("Test accuracy per class: ", acc_per_class)


if __name__ == "__main__":
    main()
