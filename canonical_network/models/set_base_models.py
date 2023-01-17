from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as tmf
import wandb
import torch_scatter as ts
#import torchsort
import dgl


class SequentialMultiple(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# Set model base class


class BaseSetModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.num_embeddings = hyperparams.num_embeddings
        self.num_layers = hyperparams.num_layers
        self.hidden_dim = hyperparams.hidden_dim
        self.out_dim = hyperparams.out_dim
        self.layer_pooling = hyperparams.layer_pooling
        self.final_pooling = hyperparams.final_pooling
        self.learning_rate = hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None

    def get_predictions(self, x):
        return x

    def training_step(self, batch, batch_idx):
        inputs, indices, targets = batch

        output = self(inputs, indices, batch_idx)
        predictions = self.get_predictions(output).squeeze()

        loss = F.binary_cross_entropy(predictions, targets.to(torch.float32))
        accuracy = tmf.accuracy(predictions, targets)

        metrics = {"train/loss": loss, "train/accuracy": accuracy}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, indices, targets = batch

        output = self(inputs, indices, batch_idx)
        predictions = self.get_predictions(output)

        loss = F.binary_cross_entropy(predictions.squeeze(), targets.to(torch.float32).squeeze())
        accuracy = tmf.accuracy(predictions, targets)
        f1_score = tmf.f1_score(predictions, targets)

        if self.global_step == 0:
            wandb.define_metric("valid/loss", summary="min")
            wandb.define_metric("valid/accuracy", summary="max")
            wandb.define_metric("valid/f1_score", summary="max")

        metrics = {"valid/loss": loss, "valid/accuracy": accuracy, "valid/f1_score": f1_score}
        self.log_dict(metrics, prog_bar=True)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-6, mode="max")
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/f1_score"}

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])

        predictions = validation_step_outputs[0]
        if self.current_epoch == 0:
            model_filename = (
                f"canonical_network/results/digits/onnx_models/{self.model}_{wandb.run.name}_{str(self.global_step)}.onnx"
            )
            # torch.onnx.export(self, (self.dummy_input, self.dummy_indices, 0.0), model_filename, opset_version=12)
            wandb.save(model_filename)

        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(predictions.to("cpu")),
                "global_step": self.global_step,
            }
        )

class ClassificationSetModel(BaseSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def training_step(self, batch, batch_idx):
        inputs, indices, targets = batch

        output = self(inputs, indices, batch_idx)
        predictions = self.get_predictions(output).squeeze()

        loss = F.cross_entropy(predictions.squeeze(), targets.squeeze())
        accuracy = tmf.accuracy(predictions, targets, task='multiclass', num_classes=10)

        metrics = {"train/loss": loss, "train/accuracy": accuracy}
        self.log_dict(metrics, on_epoch=True, batch_size=64)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, indices, targets = batch

        output = self(inputs, indices, batch_idx)
        predictions = self.get_predictions(output)

        loss = F.cross_entropy(predictions.squeeze(), targets.squeeze())
        accuracy = tmf.accuracy( predictions, targets, task='multiclass', num_classes=10)  
        # f1_score = tmf.f1_score(predictions, targets)

        if self.global_step == 0:
            wandb.define_metric("valid/loss", summary="min")
            wandb.define_metric("valid/accuracy", summary="max")
            # wandb.define_metric("valid/f1_score", summary="max")

        metrics = {"valid/loss": loss, "valid/accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True, batch_size=64)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-6, mode="max")
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/accuracy"}


# DeepSets model

class SetLayer(pl.LightningModule):
    def __init__(self, in_dim, out_dim, pooling="sum"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pooling = pooling

        self.identity_linear = nn.Linear(in_dim, out_dim)
        self.pooling_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, set_indices):
        identity = self.identity_linear(x)

        pooled_set = ts.scatter(x, set_indices, 0, reduce=self.pooling)  / 100
        pooling = self.pooling_linear(pooled_set)
        pooling = torch.index_select(pooling, 0, set_indices)

        output = F.relu(identity + pooling) + x

        return output, set_indices


class DeepSets(ClassificationSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = "deepsets"
        # self.embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.embedding_layer = nn.Linear(2, self.hidden_dim)
        self.set_layers = SequentialMultiple(
            *[SetLayer(self.hidden_dim, self.hidden_dim, self.layer_pooling) for i in range(self.num_layers - 1)]
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim) if not self.out_dim == 1 else SequentialMultiple(nn.Linear(self.hidden_dim, self.out_dim), nn.Sigmoid())

        self.dummy_input = torch.zeros(1, device=self.device, dtype=torch.long)
        self.dummy_indices = torch.zeros(1, device=self.device, dtype=torch.long)

    def forward(self, x, set_indices, _):
        embeddings = self.embedding_layer(x)
        x, _ = self.set_layers(embeddings, set_indices[:, 0])
        if self.final_pooling:
            x = ts.scatter(x, set_indices[:, 0], 0, reduce=self.final_pooling) / 100
        output = self.output_layer(x)
        return output

class CanonicalDeepSets(ClassificationSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = "canonicaldeepsets"
        self.lr = 0.1
        self.implicit = True
        self.iters = 5

        self.deepsets = DeepSets(hyperparams)
        canon_hyperparams = dict(hyperparams)
        canon_hyperparams['out_dim'] = 1
        self.energy = DeepSets(hyperparams)  # use same model for now
        self.register_buffer('initial_rotation', torch.eye(2))

    def forward(self, x, set_indices, _):
        rotated, rotation = self.min_energy(x, set_indices)
        output = self.deepsets(rotated, set_indices, _)
        # in this case, we don't necessarily care about undoing the rotation
        return output
    
    @torch.enable_grad()
    def min_energy(self, input, indices):
        # currently the optimization is being performed on the rotations directly, with gram schmidt being
        # used to project the modified matrix into a rotation again
        # alternatively, this optimization can be done on the vectors that go into gram schmidt
        real_batch_size = indices.max().item() + 1
        rotation = self.initial_rotation.clone().requires_grad_(True).unsqueeze(0).expand(real_batch_size, -1, -1)
        # print(rotation)
        for i in range(self.iters):
            if self.implicit:
                rotation = rotation.detach()
                rotation.requires_grad_(True)
            rotated = dgl.ops.gather_mm(input, rotation, idx_b=indices[:, 0])
            energy = self.energy(rotated, indices, None).sum()
            g, = torch.autograd.grad(energy, rotation, only_inputs=True, create_graph=(i == self.iters - 1) if self.implicit else True)
            rotation = rotation - self.lr * g
            rotation = self.gram_schmidt(rotation)
            print(i)
            print(rotation[0])
        rotated = dgl.ops.gather_mm(input, rotation, idx_b=indices[:, 0])
        return rotated, rotation

    def gram_schmidt(self, vectors):
        v1 = vectors[:, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = (vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        return torch.stack([v1, v2], dim=1)



# Transformer

class Transformer(BaseSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = "transformer"
        self.embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(self.hidden_dim, 8, self.hidden_dim, dropout=0,  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, self.num_layers)
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim) if not self.out_dim == 1 else SequentialMultiple(nn.Linear(self.hidden_dim, self.out_dim), nn.Sigmoid())

        self.dummy_input = torch.zeros((1,1), device=self.device, dtype=torch.long)
        self.dummy_indices = torch.zeros((1,1), device=self.device, dtype=torch.long)

    def forward(self, x, mask, _):
        embeddings = self.embedding_layer(x)
        x = self.transformer_encoder(embeddings, src_key_padding_mask=mask.float())
        # FIXME : Implement this
        # if self.final_pooling:
        #     x = ts.scatter(x, set_indices, reduce=self.final_pooling)
        output = self.output_layer(x).squeeze() * mask
        # print(output)
        return output


class Permutation(BaseSetModel):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.model = 'permutation'
        self.embedding_layer = nn.Embedding(self.num_embeddings, self.hidden_dim)
        self.project_to_score = nn.Sequential(nn.Linear(self.hidden_dim, 1))
        self.rank_to_embed = nn.Linear(1, self.hidden_dim)
        # self.model = nn.Sequential(
        #     nn.Linear(self.hidden_dim * 10, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, 10),
        # )
        self.transformer_layer = nn.TransformerEncoderLayer(self.hidden_dim, 8, self.hidden_dim, dropout=0,  batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, self.num_layers)
        self.output_layer = nn.Linear(self.hidden_dim, self.out_dim) if not self.out_dim == 1 else SequentialMultiple(nn.Linear(self.hidden_dim, self.out_dim), nn.Sigmoid())

        self.dummy_input = torch.zeros((1,10), device=self.device, dtype=torch.long)
        self.dummy_indices = torch.ones((1,10), device=self.device, dtype=torch.long)
    
    def forward(self, x, mask, _):
        embeddings = self.embedding_layer(x)
        score = self.project_to_score(embeddings).squeeze(-1)
        score = score + (1 - mask) * 1000

        # print(score.size())
        '''
        permutation = score.argsort(dim=1).unsqueeze(-1).expand_as(embeddings)
        sorted = embeddings.gather(1, permutation)

        # print(score)
        rank = torchsort.soft_rank(score) - 1
        # print(rank)
        # print(mask)
        rank = rank.unsqueeze(-1).expand_as(embeddings)
        left_idx = rank.long()
        right_idx = torch.min((left_idx + 1), mask.sum(dim=1, keepdim=True).unsqueeze(-1) - 1)
        frac = rank.frac()
        left = embeddings.gather(1, left_idx)
        right = embeddings.gather(1, right_idx)
        soft_sorted = (1 - frac) * left + frac * right

        x = sorted + (soft_sorted - soft_sorted.detach())
        x = x * mask.unsqueeze(-1)
        '''

        rank = torchsort.soft_rank(score)
        embeddings = embeddings + self.rank_to_embed(rank.unsqueeze(-1))
        
        x = self.transformer_encoder(embeddings, src_key_padding_mask=mask.float())
        output = self.output_layer(x).squeeze() * mask
        return output
