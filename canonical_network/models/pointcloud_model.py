from gettext import translation
from sched import scheduler
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import torchmetrics.functional as tmf
import wandb
from canonical_network.models.pointcloud_base_models import DGCNN

from models.vn_layers import *
from models.pointcloud_base_models import BasePointcloudModel, Pointnet, VNPointnetSmall, DGCNN
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
    "rotation": "aligned",
    "num_parts": 50,
    "num_classes": 16,
    "n_knn": 40,
    "num_points": 2048,
}

class PointcloudCanonFunction(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.n_knn = hyperparams.n_knn
        self.normal_channel = hyperparams.normal_channel
        self.num_parts = hyperparams.num_parts
        self.num_classes = hyperparams.num_classes
        self.num_points = hyperparams.num_points
        self.pooling = hyperparams.pooling
        self.rotation = hyperparams.rotation
        self.model_type = hyperparams.canon_model_type

        model_hyperparams = {
            "n_knn": self.n_knn,
            "normal_channel": self.normal_channel,
            "num_parts": self.num_parts,
            "num_classes": self.num_classes,
            "pooling": self.pooling,
            "rotation": self.rotation,
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


class PointcloudPredFunction(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.n_knn = hyperparams.n_knn
        self.normal_channel = hyperparams.normal_channel
        self.num_parts = hyperparams.num_parts
        self.num_classes = hyperparams.num_classes
        self.num_points = hyperparams.num_points
        self.pooling = hyperparams.pooling
        self.rotation = hyperparams.rotation
        self.regularization_transform = hyperparams.regularization_transform    
        self.model_type = hyperparams.pred_model_type

        model_hyperparams = {
            "n_knn": self.n_knn,
            "normal_channel": self.normal_channel,
            "num_parts": self.num_parts,
            "num_classes": self.num_classes,
            "num_points": self.num_points,
            "pooling": self.pooling,
            "rotation": self.rotation,
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
        canonical_point_cloud = torch.bmm(point_cloud.transpose(1,2), rotation_matrix_inverse) - translation_vectors
        canonical_point_cloud = canonical_point_cloud.transpose(1,2)

        return self.pred_function(canonical_point_cloud, label)[0], rotation_matrix

    def get_predictions(self, outputs):
        if type(outputs) == list:
            outputs = list(zip(*outputs))
            return torch.stack(outputs[0])
        return outputs[0]
