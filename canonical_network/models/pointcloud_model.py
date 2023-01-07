import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import torchmetrics.functional as tmf
import wandb

from canonical_network.models.vn_layers import *
from canonical_network.models.pointcloud_base_models import BasePointcloudModel, Pointnet, VNPointnetSmall, DGCNN
from canonical_network.utils import define_hyperparams, dict_to_object

SEGMENTATION_CLASSES = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}

SHAPENET_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "pooling": "mean",
    "normal_channel": False,
    "regularization_transform": 0,
    "train_rotation": "z",
    "valid_rotation": "z",
    "num_parts": 50,
    "num_classes": 16,
    "n_knn": 40,
    "num_points": 2048,
}


class PointcloudOptimizationFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        self.model_type = hyperparams.canon_model_type
        self.lr = hyperparams.lr
        self.implicit = hyperparams.implicit
        self.energy = ...  # takes point cloud as input and returns scalar energy
        self.register_buffer('initial_rotation', torch.eye(3))

    def min_energy(self, input):
        # currently the optimization is being performed on the rotations directly, with gram schmidt being
        # used to project the modified matrix into a rotation again
        # alternatively, this optimization can be done on the vectors that go into gram schmidt
        rotation = self.initial_rotation.unsqueeze(0).expand(input.size(0), -1, -1)
        for i in range(5):
            if self.implicit:
                rotation = rotation.detach()
            rotated = torch.einsum('ncl, ncd -> nld', input, rotation)  # TODO verify that input shape is (batch size, channels, num_elements)
            energy = self.energy(rotated)
            g, = torch.autograd.grad(energy, rotation, only_inputs=True, create_graph=True)
            rotation = rotation - self.lr * g
            rotation = self.gram_schmidt(rotation)
        return rotation

    def forward(self, points, labels):
        rotation_to_apply = self.min_energy(points)
        return rotation_to_apply.transpose(1, 2)  # compatibility with PointcloudCanonFunction

    def gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True).clamp(min=1e-20)
        v2 = (vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True).clamp(min=1e-20)
        v3 = (vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1 - torch.sum(vectors[:, 2] * v2, dim=1, keepdim=True) * v2)
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True).clamp(min=1e-20)
        return torch.stack([v1, v2, v3], dim=1)


class PointcloudCanonFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.n_knn = hyperparams.n_knn
        self.normal_channel = hyperparams.normal_channel
        self.num_parts = hyperparams.num_parts
        self.num_classes = hyperparams.num_classes
        self.num_points = hyperparams.num_points
        self.pooling = hyperparams.pooling
        self.train_rotation = hyperparams.train_rotation
        self.valid_rotation = hyperparams.valid_rotation
        self.model_type = hyperparams.canon_model_type

        model_hyperparams = {
            "n_knn": self.n_knn,
            "normal_channel": self.normal_channel,
            "num_parts": self.num_parts,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "train_rotation": self.train_rotation,
            "valid_rotation": self.valid_rotation,
            "num_points": self.num_points,
        }

        self.model = {"vn_pointnet": lambda: VNPointnetSmall(define_hyperparams(model_hyperparams))}[self.model_type]()
    
    def forward(self, points, labels):
        vectors = self.model(points, labels)
        rotation_vectors = vectors[:, :3]
        translation_vectors = vectors[:, 3:]

        rotation_matrix = self.gram_schmidt(rotation_vectors)
        return rotation_matrix, translation_vectors
    
    def gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = (vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = (vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1 - torch.sum(vectors[:, 2] * v2, dim=1, keepdim=True) * v2)
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)


class PointcloudPredFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.n_knn = hyperparams.n_knn
        self.normal_channel = hyperparams.normal_channel
        self.num_parts = hyperparams.num_parts
        self.num_classes = hyperparams.num_classes
        self.num_points = hyperparams.num_points
        self.pooling = hyperparams.pooling
        self.train_rotation = hyperparams.train_rotation
        self.valid_rotation = hyperparams.valid_rotation
        self.regularization_transform = hyperparams.regularization_transform    
        self.model_type = hyperparams.pred_model_type

        model_hyperparams = {
            "n_knn": self.n_knn,
            "normal_channel": self.normal_channel,
            "num_parts": self.num_parts,
            "num_classes": self.num_classes,
            "num_points": self.num_points,
            "pooling": self.pooling,
            "train_rotation": self.train_rotation,
            "valid_rotation": self.valid_rotation,
            "regularization_transform": self.regularization_transform,
        }

        self.model = {"pointnet": lambda: Pointnet(define_hyperparams(model_hyperparams)), "DGCNN": lambda: DGCNN(define_hyperparams(model_hyperparams))}[self.model_type]()
    
    def forward(self, points, labels):
        return self.model(points, labels)


class PointcloudModel(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(PointcloudModel, self).__init__(hyperparams)
        self.model = "pointcloud_model"
        self.hyperparams = hyperparams
        self.num_parts = hyperparams.num_parts
        self.normal_channel = hyperparams.normal_channel
        self.regularization_transform = hyperparams.regularization_transform

        self.canon_function = PointcloudCanonFunction(hyperparams)
        self.pred_function = PointcloudPredFunction(hyperparams)

    def forward(self, point_cloud, label):
        rotation_matrix, translation_vectors = self.canon_function(point_cloud, label)
        rotation_matrix_inverse = rotation_matrix.transpose(1,2)

        # not applying translations
        canonical_point_cloud = torch.bmm(point_cloud.transpose(1,2), rotation_matrix_inverse)
        canonical_point_cloud = canonical_point_cloud.transpose(1,2)

        return self.pred_function(canonical_point_cloud, label)[0], rotation_matrix

    def get_predictions(self, outputs):
        if type(outputs) == list:
            outputs = list(zip(*outputs))
            return torch.cat(outputs[0], dim=0)
        return outputs[0]
