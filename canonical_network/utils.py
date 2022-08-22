from collections import namedtuple
import pathlib

from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl


SRC_PATH = pathlib.Path(__file__).parent
DATA_PATH = SRC_PATH / "data"


def dict_to_object(dictionary):
    global Object
    print(dictionary)
    Object = namedtuple("Object", dictionary)
    out_object = Object(**dictionary)

    return out_object


def define_hyperparams(dictionary):
    global ModuleHyperparams
    ModuleHyperparams = namedtuple("ModuleHyperparams", dictionary)
    out_object = ModuleHyperparams(**dictionary)

    return out_object


class SetDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def combine_set_data_sparse(set_data):
    set_features, targets = zip(*set_data)

    set_indices = torch.LongTensor([])
    for i, elements in enumerate(set_features):
        elements_indices = torch.ones_like(elements, dtype=torch.long) * i
        set_indices = torch.cat((set_indices, elements_indices))

    batch_targets = torch.cat(targets, 0)
    batch_features = torch.cat(set_features, 0)

    return batch_features, set_indices, batch_targets
