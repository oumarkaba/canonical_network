import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import torchmetrics.functional as tmf
import wandb
import torch_scatter as ts

from canonical_network.models.gcl import E_GCL_vel, GCL
from canonical_network.models.vn_layers import VNLinearLeakyReLU, VNLinear, VNLeakyReLU, VNSoftplus
from canonical_network.models.set_base_models import SequentialMultiple
from canonical_network.models.gvp import GVP, GVPConvLayer, LayerNorm, tuple_index


class BaseEuclideangraphModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.learning_rate = hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        self.weight_decay = hyperparams.weight_decay if hasattr(hyperparams, "weight_decay") else 0.0
        self.patience = hyperparams.patience if hasattr(hyperparams, "patience") else 100
        self.edges = [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3],
        ]

        self.loss = nn.MSELoss()

        self.dummy_nodes = torch.zeros(2, 1, device=self.device, dtype=torch.float)
        self.dummy_loc = torch.zeros(2, 3, device=self.device, dtype=torch.float)
        self.dummy_edges = [
            torch.zeros(40, device=self.device, dtype=torch.long),
            torch.zeros(40, device=self.device, dtype=torch.long),
        ]
        self.dummy_vel = torch.zeros(2, 3, device=self.device, dtype=torch.float)
        self.dummy_edge_attr = torch.zeros(40, 2, device=self.device, dtype=torch.float)

    def training_step(self, batch, batch_idx):
        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]
        loc, vel, edge_attr, charges, loc_end = batch
        edges = self.get_edges(batch_size, n_nodes)

        nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties

        outputs = self(nodes, loc.detach(), edges, vel, edge_attr, charges)

        loss = self.loss(outputs, loc_end)

        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size, n_nodes, _ = batch[0].size()
        batch = [d.view(-1, d.size(2)) for d in batch]
        loc, vel, edge_attr, charges, loc_end = batch
        edges = self.get_edges(batch_size, n_nodes)

        nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties

        outputs = self(nodes, loc.detach(), edges, vel, edge_attr, charges)

        loss = self.loss(outputs, loc_end)
        if self.global_step == 0:
            wandb.define_metric("valid/loss", summary="min")

        metrics = {"valid/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-12)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience, factor=0.5, min_lr=1e-6, mode="max"
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/loss"}

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])

        # if self.current_epoch == 0:
        #     model_filename = f"canonical_network/results/nbody/onnx_models/{self.model}_{wandb.run.name}_{str(self.global_step)}.onnx"
        #     torch.onnx.export(
        #         self,
        #         (
        #             self.dummy_nodes.to(self.device),
        #             self.dummy_loc.to(self.device),
        #             [edges.to(self.device) for edges in self.dummy_edges],
        #             self.dummy_vel.to(self.device),
        #             self.dummy_edge_attr.to(self.device),
        #         ),
        #         model_filename,
        #         opset_version=12,
        #     )
        #     wandb.save(model_filename)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]).to(self.device), torch.LongTensor(self.edges[1]).to(self.device)]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


class EGNN_vel(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super(EGNN_vel, self).__init__(hyperparams)
        self.model = "EGNN"
        self.hidden_dim = hyperparams.hidden_dim
        self.in_node_nf = hyperparams.in_node_nf
        self.n_layers = 4
        self.act_fn = nn.SiLU()
        self.coords_weight = 1.0
        self.recurrent = True
        self.norm_diff = False
        self.tanh = False
        self.num_vectors = 1

        # self.reg = reg
        ### Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_dim, self.hidden_dim, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(hyperparams.in_node_nf, self.hidden_dim)
        self.add_module(
            "gcl_%d" % 0,
            E_GCL_vel(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                edges_in_d=hyperparams.in_edge_nf,
                act_fn=self.act_fn,
                coords_weight=self.coords_weight,
                recurrent=self.recurrent,
                norm_diff=self.norm_diff,
                tanh=self.tanh,
                num_vectors_out=self.num_vectors,
            ),
        )
        for i in range(1, self.n_layers - 1):
            self.add_module(
                "gcl_%d" % i,
                E_GCL_vel(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    edges_in_d=hyperparams.in_edge_nf,
                    act_fn=self.act_fn,
                    coords_weight=self.coords_weight,
                    recurrent=self.recurrent,
                    norm_diff=self.norm_diff,
                    tanh=self.tanh,
                    num_vectors_in=self.num_vectors,
                    num_vectors_out=self.num_vectors,
                ),
            )
        self.add_module(
            "gcl_%d" % (self.n_layers - 1),
            E_GCL_vel(
                self.hidden_dim,
                self.hidden_dim,
                self.hidden_dim,
                edges_in_d=hyperparams.in_edge_nf,
                act_fn=self.act_fn,
                coords_weight=self.coords_weight,
                recurrent=self.recurrent,
                norm_diff=self.norm_diff,
                tanh=self.tanh,
                num_vectors_in=self.num_vectors,
                last_layer=True,
            ),
        )

    def forward(self, h, x, edges, vel, edge_attr, _):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, vel, edge_attr=edge_attr)
        return x.squeeze(2)


class GNN(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super(GNN, self).__init__(hyperparams)
        self.model = "GNN"
        self.hidden_dim = hyperparams.hidden_dim
        self.input_dim = hyperparams.input_dim
        self.n_layers = hyperparams.num_layers
        self.act_fn = nn.SiLU()
        self.attention = 0
        self.recurrent = True
        ### Encoder
        # self.add_module("gcl_0", GCL(self.hidden_dim, self.hidden_dim, self.hidden_dim, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, self.n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    edges_in_nf=2,
                    act_fn=self.act_fn,
                    attention=self.attention,
                    recurrent=self.recurrent,
                ),
            )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), self.act_fn, nn.Linear(self.hidden_dim, 3)
        )
        self.embedding = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim))

    def forward(self, nodes, loc, edges, vel, edge_attr, _):
        nodes = torch.cat([loc, vel], dim=1)
        h = self.embedding(nodes)
        # h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)
        for i in range(0, self.n_layers): 
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        # return h
        return self.decoder(h)


class VNDeepSets(BaseEuclideangraphModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.prediction_mode = hyperparams.out_dim == 1
        self.model = "vndeepsets"
        self.hidden_dim = hyperparams.hidden_dim
        self.layer_pooling = hyperparams.layer_pooling
        self.final_pooling = hyperparams.final_pooling
        self.num_layers = hyperparams.num_layers
        self.nonlinearity = hyperparams.nonlinearity
        self.canon_feature = hyperparams.canon_feature
        self.canon_translation = hyperparams.canon_translation
        self.angular_feature = hyperparams.angular_feature
        self.dropout = hyperparams.dropout
        self.out_dim = hyperparams.out_dim
        self.in_dim = len(self.canon_feature)
        self.first_set_layer = VNDeepSetLayer(
            self.in_dim, self.hidden_dim, self.nonlinearity, self.layer_pooling, False, dropout=self.dropout
        )
        self.set_layers = SequentialMultiple(
            *[
                VNDeepSetLayer(
                    self.hidden_dim, self.hidden_dim, self.nonlinearity, self.layer_pooling, dropout=self.dropout
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.output_layer = (
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.batch_size = hyperparams.batch_size

        self.dummy_input = torch.zeros(1, device=self.device, dtype=torch.long)
        self.dummy_indices = torch.zeros(1, device=self.device, dtype=torch.long)

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        batch_indices = torch.arange(self.batch_size, device=self.device).reshape(-1, 1)
        batch_indices = batch_indices.repeat(1, 5).reshape(-1)
        mean_loc = ts.scatter(loc, batch_indices, 0, reduce=self.layer_pooling)
        mean_loc = mean_loc.repeat(5, 1, 1).transpose(0, 1).reshape(-1, 3)
        canonical_loc = loc - mean_loc
        if self.canon_feature == "p":
            features = torch.stack([canonical_loc], dim=2)
        if self.canon_feature == "pv":
            features = torch.stack([canonical_loc, vel], dim=2)
        elif self.canon_feature == "pva":
            angular = torch.linalg.cross(canonical_loc, vel, dim=1)
            features = torch.stack([canonical_loc, vel, angular], dim=2)
        elif self.canon_feature == "pvc":
            features = torch.stack([canonical_loc, vel, canonical_loc * charges], dim=2)
        elif self.canon_feature == "pvac":
            angular = torch.linalg.cross(canonical_loc, vel, dim=1)
            features = torch.stack([canonical_loc, vel, angular, canonical_loc * charges], dim=2)

        x, _ = self.first_set_layer(features, edges)
        x, _ = self.set_layers(x, edges)

        if self.prediction_mode:
            output = self.output_layer(x)
            output = output.squeeze()
            return output
        else:
            x = ts.scatter(x, batch_indices, 0, reduce=self.final_pooling)
        output = self.output_layer(x)

        output = output.repeat(5, 1, 1, 1).transpose(0, 1)
        output = output.reshape(-1, 3, 4)

        rotation_vectors = output[:, :, :3]
        translation_vectors = output[:, :, 3:] if self.canon_translation else 0.0
        translation_vectors = translation_vectors + mean_loc[:, :, None]

        return rotation_vectors, translation_vectors.squeeze()


class VNDeepSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity, pooling="sum", residual=True, dropout=0.0):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels
        self.pooling = pooling
        self.residual = residual
        self.nonlinearity = nonlinearity
        self.dropout = dropout

        self.identity_linear = nn.Linear(in_channels, out_channels)
        self.pooling_linear = nn.Linear(in_channels, out_channels)

        self.dropout_layer = nn.Dropout(self.dropout)

        if self.nonlinearity == "softplus":
            self.nonlinear_function = VNSoftplus(out_channels, share_nonlinearity=False)
        elif self.nonlinearity == "relu":
            self.nonlinear_function = VNLeakyReLU(out_channels, share_nonlinearity=False, negative_slope=0.0)
        elif self.nonlinearity == "leakyrelu":
            self.nonlinear_function = VNLeakyReLU(out_channels, share_nonlinearity=False)

    def forward(self, x, edges):
        edges_1 = edges[0]
        edges_2 = edges[1]

        identity = self.identity_linear(x)

        nodes_1 = torch.index_select(x, 0, edges_1)
        pooled_set = ts.scatter(nodes_1, edges_2, 0, reduce=self.pooling)
        pooling = self.pooling_linear(pooled_set)

        output = self.nonlinear_function((identity + pooling).transpose(1, -1)).transpose(1, -1)

        output = self.dropout_layer(output)

        if self.residual:
            output = output + x

        return output, edges

class GVP_GNN(BaseEuclideangraphModel):
    '''
    GVP-GNN for structure-conditioned autoregressive 
    protein design as described in manuscript.
    '''
    def __init__(self, hyperparams):
        self.node_in_dim = (1, 2)
        self.hidden_dim = hyperparams.hidden_dim
        self.node_h_dim = (hyperparams.hidden_dim, hyperparams.hidden_dim)
        self.edge_in_dim = (2, 0)
        self.edge_h_dim = (hyperparams.hidden_dim, hyperparams.hidden_dim)
        self.out_dim = hyperparams.out_dim
        self.num_layers = hyperparams.num_layers
        self.drop_rate = hyperparams.dropout
        self.final_pooling = hyperparams.final_pooling
        self.canon_translation = hyperparams.canon_translation
        self.batch_size = hyperparams.batch_size
        self.layer_pooling = hyperparams.layer_pooling
    
        super(GVP_GNN, self).__init__(hyperparams)
        
        self.W_v = nn.Sequential(
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None)),
            LayerNorm(self.node_h_dim)
        )
        self.W_e = nn.Sequential(
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None)),
            LayerNorm(self.edge_h_dim)
        )
        
        self.encoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, self.edge_h_dim, drop_rate=self.drop_rate) 
            for _ in range(self.num_layers))
      
        self.last_layer = GVPConvLayer(self.node_h_dim, self.edge_h_dim, drop_rate=self.drop_rate)
        self.output_layer = (
            nn.Linear(self.hidden_dim, self.out_dim)
        )

    def forward(self, nodes, loc, edges, vel, edge_attr, charges):
        '''
        Forward pass to be used at train-time, or evaluating likelihood.
        
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: int `torch.Tensor` of shape [num_nodes]
        '''
        batch_indices = torch.arange(self.batch_size, device=self.device).reshape(-1, 1)
        batch_indices = batch_indices.repeat(1, 5).reshape(-1)
        mean_loc = ts.scatter(loc, batch_indices, 0, reduce=self.layer_pooling)
        mean_loc = mean_loc.repeat(5, 1, 1).transpose(0, 1).reshape(-1, 3)
        canonical_loc = loc - mean_loc

        h_V = (nodes, torch.stack([canonical_loc, vel], dim=1))
        h_E = edge_attr
        edges = torch.stack(edges, dim=0)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edges, h_E)
        
        h_V = self.last_layer(h_V, edges, h_E)[1]
        
        x = ts.scatter(h_V, batch_indices, 0, reduce=self.final_pooling)
        x = x.transpose(1, 2)
        
        output = self.output_layer(x)

        output = output.repeat(5, 1, 1, 1).transpose(0, 1)
        output = output.reshape(-1, 3, 4)

        rotation_vectors = output[:, :, :3]
        translation_vectors = output[:, :, 3:] if self.canon_translation else 0.0
        translation_vectors = translation_vectors + mean_loc[:, :, None]

        return rotation_vectors, translation_vectors.squeeze()

