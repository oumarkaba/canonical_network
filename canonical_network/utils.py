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


def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y]

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def random_shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    # generate random shifts in pytorch from -shift_range to shift_range of shape (B, 3)
    shifts = torch.rand((B, 3), device=batch_data.device) * 2 * shift_range - shift_range
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    # generate random shifts in pytorch from -scale_low to scale_high of shape (B) and put in on the device of batch_data
    scales = torch.rand((B,), device=batch_data.device) * (scale_high - scale_low) + scale_low
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  torch.rand()*max_dropout_ratio # 0~0.875
        drop_idx = torch.where(torch.rand((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None:  # dynamic knn graph
            idx = knn(x, k=k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    idx_base = torch.arange(0, batch_size).type_as(idx).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size).type_as(idx).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

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