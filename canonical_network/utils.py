from collections import namedtuple
import pathlib

from torch.utils.data import Dataset
import torch
import torchvision
import numpy as np
import kornia as K
import os


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

def save_images_class_wise(images, labels, save_path, filename, num_classes=10):
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    for i in range(num_classes):
        images_class = images[labels == i]
        torchvision.utils.save_image(images_class, save_path + '/' +  f"{filename}_class_{i}.png", nrow=10)

def check_rotation_invariance(network, x, num_rotations=4):
    batch_size = x.shape[0]
    device = x.device
    x_out = network(x).argmax(dim=-1)
    angles = torch.linspace(0., 360., steps=num_rotations + 1, dtype=torch.float32)[:num_rotations].to(device)
    truth = True
    for i, angle in enumerate(angles):
        angle_batch = angle * torch.ones(batch_size).to(device)
        x_rotated = K.geometry.rotate(x, angle_batch)
        x_rotated_out = network(x_rotated).argmax(dim=-1)
        cur_truth = np.allclose(x_rotated_out.detach().cpu().numpy(), x_out.detach().cpu().numpy(), atol=1e-1)
        truth = truth and cur_truth
    return 1.0 if truth else 0.0

def check_rotoreflection_invariance(network, x, num_rotations=4):
    batch_size = x.shape[0]
    device = x.device
    x_out = network(x).argmax(dim=-1)
    angles = torch.linspace(0., 360., steps=num_rotations + 1, dtype=torch.float32)[:num_rotations].to(device)
    truth = True
    for i, angle in enumerate(angles):
        angle_batch = angle * torch.ones(batch_size).to(device)
        x_rotated = K.geometry.rotate(x, angle_batch)
        x_rotated_out = network(x_rotated).argmax(dim=-1)
        cur_truth = np.allclose(x_rotated_out.detach().cpu().numpy(), x_out.detach().cpu().numpy(), atol=1e-1)
        truth = truth and cur_truth
    for i, angle in enumerate(angles):
        angle_batch = angle * torch.ones(batch_size).to(device)
        x_rotated = K.geometry.rotate(x, angle_batch)
        x_rotated_reflected = K.geometry.hflip(x_rotated)
        x_rotated_reflected_out = network(x_rotated_reflected).argmax(dim=-1)
        cur_truth = np.allclose(x_rotated_reflected_out.detach().cpu().numpy(), x_out.detach().cpu().numpy(), atol=1e-1)
        truth = truth and cur_truth
    return 1.0 if truth else 0.0


def check_rotation_equivariance(network, x, num_rotations=4):
    batch_size = x.shape[0]
    device = x.device
    x_out = network(x)
    print('Shape of the output is ', x_out.shape)
    angles = torch.linspace(0., 360., steps=num_rotations + 1, dtype=torch.float32)[:num_rotations].to(device)
    for i, angle in enumerate(angles):
        angle_batch = angle * torch.ones(batch_size).to(device)
        x_rotated = K.geometry.rotate(x, angle_batch)
        x_rotated_out = network(x_rotated)
        print(np.allclose(x_rotated_out.mean(dim=(1, 3, 4)).detach().cpu().numpy(),
                          x_out.mean(dim=(1, 3, 4)).roll(i, 1).detach().cpu().numpy(), atol=1e-6))


def check_rotoreflection_equivariance(network, x, num_rotations=4):
    batch_size = x.shape[0]
    device = x.device
    x_out = network(x)
    print('Shape of the output is ', x_out.shape)
    angles = torch.linspace(0., 360., steps=num_rotations + 1, dtype=torch.float32)[:num_rotations].to(device)
    for i, angle in enumerate(angles):
        angle_batch = angle * torch.ones(batch_size).to(device)
        x_rotated = K.geometry.rotate(x, angle_batch)
        x_rotated_out = network(x_rotated)
        print(np.allclose(x_rotated_out.mean(dim=(1, 3, 4))[:, :num_rotations].detach().cpu().numpy(),
                          x_out.mean(dim=(1, 3, 4))[:, :num_rotations].roll(i, 1).detach().cpu().numpy(), atol=1e-6)
         and np.allclose(x_rotated_out.mean(dim=(1, 3, 4))[:, num_rotations:].detach().cpu().numpy(),
                          x_out.mean(dim=(1, 3, 4))[:, num_rotations:].roll(-i, 1).detach().cpu().numpy(), atol=1e-6))
    for i, angle in enumerate(angles):
        angle_batch = angle * torch.ones(batch_size).to(device)
        x_rotated = K.geometry.rotate(x, angle_batch)
        x_rotated_reflected = K.geometry.hflip(x_rotated)
        x_rotated_reflected_out = network(x_rotated_reflected)
        print(np.allclose(x_rotated_reflected_out.mean(dim=(1, 3, 4))[:, :num_rotations].detach().cpu().numpy(),
                          x_out.mean(dim=(1, 3, 4))[:, num_rotations:].roll(-i, 1).detach().cpu().numpy(), atol=1e-6)
         and np.allclose(x_rotated_reflected_out.mean(dim=(1, 3, 4))[:, num_rotations:].detach().cpu().numpy(),
                         x_out.mean(dim=(1, 3, 4))[:, :num_rotations].roll(i, 1).detach().cpu().numpy(), atol=1e-6))