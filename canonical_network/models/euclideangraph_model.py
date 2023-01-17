import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import torchmetrics.functional as tmf
import wandb

from canonical_network.models.vn_layers import *
from canonical_network.models.euclideangraph_base_models import EGNN_vel, GNN, VNDeepSets, BaseEuclideangraphModel
from canonical_network.utils import define_hyperparams, dict_to_object

NBODY_HYPERPARAMS = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-12,
    "patience": 10000,
    "hidden_nf": 64,
    "input_dim": 6,
    "in_node_nf": 1,
    "in_edge_nf": 2,
    "num_layers": 4,
    "canon_num_layers": 4,
    "canon_hidden_dim": 16,
    "canon_layer_pooling": "mean",
    "canon_final_pooling": "mean",
    "canon_nonlinearity": "relu",
    "canon_feature": "pv",
    "canon_translation": False,
    "canon_angular_feature": 0,
    "canon_dropout": 0.5,
    "freeze_canon": False,
}


class EuclideangraphCanonFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.canon_model_type
        self.num_layers = hyperparams.canon_num_layers
        self.hidden_dim = hyperparams.canon_hidden_dim
        self.layer_pooling = hyperparams.canon_layer_pooling
        self.final_pooling = hyperparams.canon_final_pooling
        self.nonlinearity = hyperparams.canon_nonlinearity
        self.canon_feature = hyperparams.canon_feature
        self.angular_feature = hyperparams.canon_angular_feature
        self.dropout = hyperparams.canon_dropout
        self.batch_size = hyperparams.batch_size
        self.canon_translation = hyperparams.canon_translation

        model_hyperparams = {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "layer_pooling": self.layer_pooling,
            "final_pooling": self.final_pooling,
            "out_dim": 4,
            "batch_size": self.batch_size,
            "nonlinearity": self.nonlinearity,
            "canon_feature": self.canon_feature,
            "canon_translation": self.canon_translation,
            "angular_feature": self.angular_feature,
            "dropout": self.dropout,
        }

        self.model = {
            "EGNN": lambda: EGNN_vel(define_hyperparams(model_hyperparams)),
            "vndeepsets": lambda: VNDeepSets(define_hyperparams(model_hyperparams)),
        }[self.model_type]()

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        rotation_vectors, translation_vectors = self.model(nodes, loc, edges, vel, edge_attr, charges)

        rotation_matrix = self.modified_gram_schmidt(rotation_vectors)

        return rotation_matrix, translation_vectors

    def gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = (
            vectors[:, 2]
            - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
            - torch.sum(vectors[:, 2] * v2, dim=1, keepdim=True) * v2
        )
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)

    def modified_gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1
        v3 = v3 - torch.sum(v3 * v2, dim=1, keepdim=True) * v2
        v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
        return torch.stack([v1, v2, v3], dim=1)


class EuclideangraphPredFunction(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.pred_model_type
        self.num_layers = hyperparams.num_layers
        self.hidden_nf = hyperparams.hidden_nf
        self.input_dim = hyperparams.input_dim
        self.in_node_nf = hyperparams.in_node_nf
        self.in_edge_nf = hyperparams.in_edge_nf

        model_hyperparams = {
            "num_layers": self.num_layers,
            "hidden_nf": self.hidden_nf,
            "input_dim": self.input_dim,
            "in_node_nf": self.in_node_nf,
            "in_edge_nf": self.in_edge_nf,
        }

        self.model = {
            "GNN": lambda: GNN(define_hyperparams(model_hyperparams)),
            "EGNN": lambda: EGNN_vel(define_hyperparams(model_hyperparams)),
        }[self.model_type]()

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        return self.model(nodes, loc, edges, vel, edge_attr, charges)


class EuclideanGraphModel(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super(EuclideanGraphModel, self).__init__(hyperparams)
        self.model = "euclideangraph_model"
        self.hyperparams = hyperparams
        self.canon_translation = hyperparams.canon_translation

        self.canon_function = EuclideangraphCanonFunction(hyperparams)
        self.pred_function = EuclideangraphPredFunction(hyperparams)

        if hyperparams.freeze_canon:
            self.canon_function.freeze()

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        rotation_matrix, translation_vectors = self.canon_function(nodes, loc, edges, vel, edge_attr, charges)
        rotation_matrix_inverse = rotation_matrix.transpose(1, 2)

        canonical_loc = (
            torch.bmm(loc[:, None, :], rotation_matrix_inverse).squeeze()
            - torch.bmm(translation_vectors[:, None, :], rotation_matrix_inverse).squeeze()
        )
        canonical_vel = torch.bmm(vel[:, None, :], rotation_matrix_inverse).squeeze()

        position_prediction = self.pred_function(nodes, canonical_loc, edges, canonical_vel, edge_attr, charges)

        position_prediction = (
            torch.bmm(position_prediction[:, None, :], rotation_matrix).squeeze() + translation_vectors
        )

        return position_prediction
