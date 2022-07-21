import numpy as np
from numpy.random import RandomState
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import pytorch_lightning as pl

import canonical_network.utils as utils

RNG = RandomState(0)


class DigitsGenerator:
    def __init__(self, max_number, max_size, task):
        self.max_number = max_number
        self.max_size = max_size
        self.task = task

    def generate_numbers(self):
        random_ints = RNG.randint(1, self.max_number, self.max_size)
        size = RNG.randint(1, self.max_size)
        random_ints = random_ints[:size]

        return random_ints

    def generate_targets(self, numbers):
        if self.task == "sum":
            set_quantity = np.sum(numbers)
        elif self.task == "max":
            set_quantity = np.max(numbers)
        else:
            raise ValueError("Task not defined")

        modulo = set_quantity % numbers
        targets = modulo == 0

        return targets

    def generate_dataset_set(self, num_samples):
        samples = []
        all_targets = np.array([])
        lenghts = np.array([])
        for i in range(num_samples):
            numbers = self.generate_numbers()
            targets = self.generate_targets(numbers)
            if i % 100 == 0:
                print(f"generating sample {i}")
                print(numbers)
            all_targets = np.append(all_targets, targets)
            lenghts = np.append(lenghts, numbers.shape)

            numbers_tensor = torch.LongTensor(numbers)
            targets_tensor = torch.LongTensor(targets)

            samples.append((numbers_tensor, targets_tensor))
        print(np.mean(all_targets))

        return samples

    def generate_dataset_pad(self, num_samples):
        samples_matrix = np.zeros((num_samples, self.max_number))
        targets_matrix = np.zeros((num_samples, self.max_number))
        mask_matrix = np.zeros((num_samples, self.max_number))
        for i in range(num_samples):
            numbers = self.generate_numbers()
            targets = self.generate_targets(numbers)
            if i % 100 == 0:
                print(f"generating sample {i}")
                print(numbers)
            samples_matrix[i, 0 : numbers.shape[0]] = numbers
            targets_matrix[i, 0 : numbers.shape[0]] = targets
            mask_matrix[i, 0 : numbers.shape[0]] = 1

        numbers_tensor = torch.LongTensor(samples_matrix)
        targets_tensor = torch.LongTensor(targets_matrix)
        mask_tensor = torch.LongTensor(mask_matrix)

        return numbers_tensor, mask_tensor, targets_tensor


class DigitsDataModule(pl.LightningDataModule):
    def __init__(self, hyperparams, data_path=utils.DATA_PATH / "digits"):
        super().__init__()
        self.data_path = data_path
        self.hyperparams = hyperparams
        self.mode = "pad" if hyperparams.model == "transformer" else "set"
        self.collate_fn = utils.combine_set_data_sparse if self.mode == "set" else None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = torch.load(utils.DATA_PATH / "digits" / f"train_dataset_{self.mode}.pt")
            self.valid_dataset = torch.load(utils.DATA_PATH / "digits" / f"valid_dataset_{self.mode}.pt")
        if stage == "test":
            self.test_dataset = torch.load(utils.DATA_PATH / "digits" / f"test_dataset_{self.mode}.pt")

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.hyperparams.batch_size,
            True,
            collate_fn=self.collate_fn,
            num_workers=self.hyperparams.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            self.hyperparams.batch_size,
            True,
            collate_fn=self.collate_fn,
            num_workers=self.hyperparams.num_workers,
        )
        return valid_loader


def create_digits_datasets(
    max_number, max_size, task, num_train_samples, num_valid_samples, num_test_samples, mode="set"
):
    digits_generator = DigitsGenerator(max_number, max_size, task)
    if mode == "set":
        data = digits_generator.generate_dataset_set(num_train_samples + num_valid_samples + num_test_samples)
        digits, targets = zip(*data)
        digits_dataset = utils.SetDataset(digits, targets)
    elif mode == "pad":
        numbers, mask, targets = digits_generator.generate_dataset_pad(
            num_train_samples + num_valid_samples + num_test_samples
        )
        digits_dataset = TensorDataset(numbers, mask, targets)
    else:
        raise ValueError("mode can only be set or pad")

    train_dataset, valid_dataset, test_dataset = random_split(
        digits_dataset, [num_train_samples, num_valid_samples, num_test_samples]
    )

    torch.save(train_dataset, utils.DATA_PATH / "digits" / f"train_dataset_{mode}.pt")
    torch.save(valid_dataset, utils.DATA_PATH / "digits" / f"valid_dataset_{mode}.pt")
    torch.save(test_dataset, utils.DATA_PATH / "digits" / f"test_dataset_{mode}.pt")


def main():
    create_digits_datasets(10, 10, "sum", 10000, 2000, 10000, "set")


if __name__ == "__main__":
    main()
