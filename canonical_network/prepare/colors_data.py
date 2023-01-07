import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import random_split, DataLoader, TensorDataset, RandomSampler
import pytorch_lightning as pl

import canonical_network.utils as utils

class ColorsDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, data_path=utils.DATA_PATH / "colors"):
        super().__init__()
        self.hyperparams = hyperparams
        self.data_path = data_path / self.hyperparams.dataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = torch.load(self.data_path / f"train_dataset.pt")
            self.valid_dataset = torch.load(self.data_path / f"valid_dataset.pt")
        if stage == "test":
            self.test_dataset = torch.load(self.data_path / f"test_dataset.pt")

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            True,
            num_workers=self.hyperparams.num_workers,
            collate_fn=colors_collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.batch_size,
            True,
            num_workers=self.hyperparams.num_workers,
            collate_fn=colors_collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.yesy_dataset,
            self.hyperparams.batch_size,
            True,
            num_workers=self.hyperparams.num_workers,
            collate_fn=colors_collate_fn,
        )
        return test_loader


def colors_collate_fn(batch):
    batch_images_1, batch_images_2 = zip(*batch)
    batch_images_1 = torch.stack(batch_images_1)
    batch_images_2 = torch.stack(batch_images_2)

    batch_images_1_r = batch_images_1 * torch.tensor([1, 0, 0]).view(1, 3, 1, 1)
    batch_images_1_g = batch_images_1 * torch.tensor([0, 1, 0]).view(1, 3, 1, 1)
    batch_images_1_b = batch_images_1 * torch.tensor([0, 0, 1]).view(1, 3, 1, 1)
    batch_images_2_r = batch_images_2 * torch.tensor([1, 0, 0]).view(1, 3, 1, 1)
    batch_images_2_g = batch_images_2 * torch.tensor([0, 1, 0]).view(1, 3, 1, 1)
    batch_images_2_b = batch_images_2 * torch.tensor([0, 0, 1]).view(1, 3, 1, 1)

    batch_images_set = torch.stack([batch_images_1_r, batch_images_1_g, batch_images_1_b, batch_images_2_r, batch_images_2_g, batch_images_2_b], dim=1)
    batch_targets = torch.stack([batch_images_1, batch_images_1, batch_images_1, batch_images_2, batch_images_2, batch_images_2], dim=1)

    return batch_images_set, batch_targets


def process_dataset(dataset, num_samples):
    sample_images_1 = RandomSampler(dataset, num_samples=num_samples, replacement=True)
    sample_images_2 = RandomSampler(dataset, num_samples=num_samples, replacement=True)

    tensor_sample_images_1 = torch.stack([dataset[i][0] for i in sample_images_1])
    del sample_images_1
    tensor_sample_images_2 = torch.stack([dataset[i][0] for i in sample_images_2])
    del sample_images_2

    # tensor_sample_images_1_r = tensor_sample_images_1 * torch.tensor([1, 0, 0]).view(1, 3, 1, 1)
    # tensor_sample_images_1_g = tensor_sample_images_1 * torch.tensor([0, 1, 0]).view(1, 3, 1, 1)
    # tensor_sample_images_1_b = tensor_sample_images_1 * torch.tensor([0, 0, 1]).view(1, 3, 1, 1)
    # tensor_sample_images_2_r = tensor_sample_images_2 * torch.tensor([1, 0, 0]).view(1, 3, 1, 1)
    # tensor_sample_images_2_g = tensor_sample_images_2 * torch.tensor([0, 1, 0]).view(1, 3, 1, 1)
    # tensor_sample_images_2_b = tensor_sample_images_2 * torch.tensor([0, 0, 1]).view(1, 3, 1, 1)

    # sample_images_set = torch.stack([tensor_sample_images_1_r, tensor_sample_images_1_g, tensor_sample_images_1_b, tensor_sample_images_2_r, tensor_sample_images_2_g, tensor_sample_images_2_b], dim=1)
    # sample_targets = torch.stack([tensor_sample_images_1, tensor_sample_images_1, tensor_sample_images_1, tensor_sample_images_2, tensor_sample_images_2, tensor_sample_images_2], dim=1)

    sample_images_dataset = TensorDataset(tensor_sample_images_1, tensor_sample_images_2)

    return sample_images_dataset

def create_colors_datasets():
    train_dataset = CelebA(utils.DATA_PATH / "colors" / "celeba" / "images", split="train", download=False, transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]))
    valid_dataset = CelebA(utils.DATA_PATH / "colors" / "celeba" / "images", split="valid", download=False, transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]))
    test_dataset = CelebA(utils.DATA_PATH / "colors" / "celeba" / "images", split="test", download=False, transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]))

    processed_train_dataset = process_dataset(train_dataset, num_samples=30000)
    processed_valid_dataset = process_dataset(valid_dataset, num_samples=3000)
    processed_test_dataset = process_dataset(test_dataset, num_samples=3000)

    torch.save(processed_train_dataset, utils.DATA_PATH / "colors" / "celeba" / f"train_dataset.pt")
    torch.save(processed_valid_dataset, utils.DATA_PATH / "colors" / "celeba" / f"valid_dataset.pt")
    torch.save(processed_test_dataset, utils.DATA_PATH / "colors" / "celeba" / f"test_dataset.pt")


def main():
    create_colors_datasets()


if __name__ == "__main__":
    main()
