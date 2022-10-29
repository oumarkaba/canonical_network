
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, download=False):
        super().__init__()
        self.data_path = hyperparams.data_path
        self.hyperparams = hyperparams
        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        os.makedirs(self.data_path, exist_ok=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Good strategy for splitting data
            # cifar_full = CIFAR10(self.data_path, train=True, transform=self.transform, download=True)
            # self.train_dataset, self.valid_dataset = random_split(cifar_full, [45000, 5000])
            # print('Train dataset size: ', len(self.train_dataset))
            # print('Valid dataset size: ', len(self.valid_dataset))
            # Not a good strategy for splitting data but most papers use this
            self.train_dataset = CIFAR10(self.data_path, train=True, transform=self.train_transform, download=True)
            self.valid_dataset = CIFAR10(self.data_path, train=False, transform=self.test_transform, download=True)
        if stage == "test":
            self.test_dataset = CIFAR10(self.data_path, train=False, transform=self.test_transform, download=True)
            print('Test dataset size: ', len(self.test_dataset))

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            shuffle=True,
            num_workers=self.hyperparams.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            self.hyperparams.batch_size,
            shuffle=False,
            num_workers=self.hyperparams.num_workers,
        )
        return test_loader