# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -----
# The following software may be included in this product: pytorch-unet,
# https://github.com/usuyama/pytorch-unet
# License: https://github.com/usuyama/pytorch-unet/blob/master/LICENSE
# MIT License
#
# Copyright (c) 2018 Naoto Usuyama
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torchmetrics.functional as tmf
import wandb
import torchsort

from canonical_network.models.colouring_layers import \
    SetDropout,ReluSets,Conv2dDeepSym,Conv2dSiamese,\
    SetMaxPool2d,SetUpsample,DeepSetsBlock,DeepSetsBlockSiamese,MLPBlock,Conv2dSAittala,Conv2dSridhar, Conv2d


def double_conv(in_channels, out_channels,model_type,p_drop, kernel_size=3,use_max=0, depthwise=False):
    if model_type=='deepsets' or model_type == 'fullySiamese':
        return nn.Sequential(
            Conv2dSiamese(in_channels, out_channels, kernel_size,padding=int((kernel_size-1)/2)),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSiamese(out_channels, out_channels, kernel_size,padding=int((kernel_size-1)/2)),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    if model_type=='precanonical':
        return nn.Sequential(
            Conv2dSiamese(in_channels, out_channels, kernel_size,padding=int((kernel_size-1)/2), residual=False),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSiamese(out_channels, out_channels, kernel_size,padding=int((kernel_size-1)/2), residual=True),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSiamese(out_channels, out_channels, kernel_size,padding=int((kernel_size-1)/2), residual=True),
            ReluSets(),
            SetDropout(p_drop=p_drop),
        )
    elif model_type == 'Sridhar':
        return nn.Sequential(
            Conv2dSridhar(in_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSridhar(out_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == 'Aittala':
        return nn.Sequential(
            Conv2dSAittala(in_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dSAittala(out_channels, out_channels, 3, padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == 'DeepSymmetricNet':
        return nn.Sequential(
            Conv2dDeepSym(in_channels, out_channels, 3,use_max=use_max,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop),
            Conv2dDeepSym(out_channels, out_channels, 3,use_max=use_max,padding=1),
            ReluSets(),
            SetDropout(p_drop=p_drop)
        )
    elif model_type == "canonical":
        groups = 6 if depthwise else 1
        return nn.Sequential(
            Conv2d(in_channels, out_channels, 3, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop),
            Conv2d(out_channels, out_channels, 3, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop)
        )


def last_conv(model_type,c,use_max=0):
    if model_type == 'deepsets' or model_type == 'fullySiamese':
        conv_last = Conv2dSiamese(c[2] + c[1], 3, 1,padding=0)
    elif model_type == 'Sridhar':
        conv_last = Conv2dSiamese(c[2] + c[1], 3, 1, padding=0)
    elif model_type == 'Aittala':
        conv_last = Conv2dSiamese(c[2] + c[1], 3, 1, padding=0)
    elif model_type=='DeepSymmetricNet':
        conv_last = Conv2dDeepSym(c[2] + c[1], 3, 1,use_max=use_max,padding=0)
    elif model_type == "canonical":
        conv_last = Conv2d(c[2] + c[1], 18, 1, padding=0)
    return conv_last


def get_feature_processing_block(model_type,channels):
    if model_type == 'deepsets' or model_type == 'DeepSymmetricNet'\
            or model_type == 'Sridhar' or model_type == 'Aittala':
        return DeepSetsBlock(channels=(channels[4],channels[4],channels[4]))
    elif model_type == 'canonical':
        return MLP(in_channels=channels[4], hidden_channels=(channels[4],channels[4]), norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU)
        # return DeepSetsBlock(channels=(int(channels[4]/6),int(channels[4]/6),int(channels[4]/6)))
    elif model_type == 'fullySiamese':
        return DeepSetsBlockSiamese(channels=(channels[4],3*channels[4],channels[4]))
    elif model_type == 'DeepSymmetricNetMLP':
        return MLPBlock(channels=(channels[4],channels[4],channels[4]))


class MLP(nn.Sequential):
    def __init__(self, in_channels, hidden_channels, norm_layer, activation_layer, dropout=0.0):
        super().__init__()
        params = {}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=True))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=True))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class CanonicalNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.pooling1 = torch.nn.AdaptiveAvgPool2d((32,32))
        self.cnn_block = double_conv(3, 32, "precanonical", 0.0, 5)
        self.pooling2 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.deepsets_block = DeepSetsBlock(channels=(32, 32, 32))
    
    def forward(self, x):
        b,n,c,h,w = x.size()
        x = x.view(b*n,c,h,w)
        x = self.pooling1(x)
        x = x.view(b,n,c,32,32)

        x = self.cnn_block(x)
        x = self.pooling2(x).squeeze()
        x = self.deepsets_block(x)
        score = torch.sum(x, dim=2)
        
        return score

class UNet(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.model_type = hyperparams.model_type
        self.p_drop = hyperparams.p_drop
        self.use_max = hyperparams.use_max
        self.sort_reg = hyperparams.sort_reg
        self.learning_rate = hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None
        self.patience = hyperparams.patience
        self.parameters_factor = hyperparams.parameters_factor
        self.depthwise_encoder = hyperparams.depthwise_encoder
        self.depthwise_decoder = hyperparams.depthwise_decoder
        self.weight_decay = hyperparams.weight_decay

        if self.model_type =='deepsets' or self.model_type == 'Sridhar':
            c = (3, 64, 128, 200, 300)
        elif self.model_type == 'canonical':
            c = (18, 32, 64, 96, 128)
            c = [c[0]] + [max(int(n/self.parameters_factor) * 6, 18) for n in c[1:]]
        elif self.model_type == 'Aittala':
            c = (3, 150, 200, 300, 320)
        elif self.model_type == 'DeepSymmetricNet':
            c = (3, 50, 100, 150, 200)

        if self.model_type == 'canonical':
            self.canonical_network = CanonicalNetwork()

        self.dconv_down1 = double_conv(c[0], c[1],self.model_type,self.p_drop,use_max=self.use_max, depthwise=self.depthwise_encoder)
        self.dconv_down2 = double_conv(c[1], c[2],self.model_type,self.p_drop,use_max=self.use_max, depthwise=self.depthwise_encoder)
        self.dconv_down3 = double_conv(c[2], c[3],self.model_type,self.p_drop,use_max=self.use_max, depthwise=self.depthwise_encoder)
        self.dconv_down4 = double_conv(c[3], c[4],self.model_type,self.p_drop,use_max=self.use_max, depthwise=self.depthwise_encoder)

        self.maxpool2 = nn.MaxPool2d(kernel_size = 2,stride=2) if self.model_type == "canonical" else SetMaxPool2d(stride=2)
        self.maxpool8 = nn.MaxPool2d(kernel_size = 2,stride=8) if self.model_type == "canonical" else SetMaxPool2d(stride=8)

        self.feature_processing_block = get_feature_processing_block(model_type=self.model_type,channels=c)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if self.model_type == "canonical" else SetUpsample(scale_factor=2)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) if self.model_type == "canonical" else SetUpsample(scale_factor=8)

        self.dconv_up3 = double_conv(c[4] + c[4], c[3],self.model_type,self.p_drop,use_max=self.use_max, depthwise=self.depthwise_decoder)
        self.dconv_up2 = double_conv(c[3] + c[3], c[2],self.model_type,self.p_drop,use_max=self.use_max, depthwise=self.depthwise_decoder)
        self.dconv_up1 = double_conv(c[2] + c[2], c[2],self.model_type,self.p_drop,use_max=self.use_max, depthwise=self.depthwise_decoder)

        self.last_conv = last_conv(self.model_type,c,use_max=self.use_max)


    def forward(self, x):
        if self.model_type == 'canonical':
            # score = self.canonical_network(x)
            # x = self.sort(x, score)
            x = x.view(-1, 18, 64, 64)
            
        conv1 = self.dconv_down1(x)
        x = self.maxpool2(conv1) #32

        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2) #16

        conv3 = self.dconv_down3(x)
        x = self.maxpool2(conv3) #8

        conv4 = self.dconv_down4(x)
        x = self.maxpool8(conv4)

        # deep sets block

        axis_correction = 0 if self.model_type == "canonical" else 1

        x = x.squeeze()
        x = self.feature_processing_block(x)

        x = x.unsqueeze(dim=2+axis_correction).unsqueeze(dim=3+axis_correction)

        x = self.upsample8(x)#8
        x = torch.cat([x, conv4], dim=1+axis_correction)

        x = self.dconv_up3(x)
        x = self.upsample2(x)#16
        x = torch.cat([x, conv3], dim=1+axis_correction)

        x = self.dconv_up2(x)
        x = self.upsample2(x)#32
        x = torch.cat([x, conv2], dim=1+axis_correction)

        x = self.dconv_up1(x)
        x = self.upsample2(x)#64
        x = torch.cat([x, conv1], dim=1+axis_correction)

        out = self.last_conv(x)

        if self.model_type == 'canonical':
            out = out.view(-1, 6, 3, 64, 64)
            # out = self.unsort(out, score)

        return out
    
    def sort(self, x, score):
        permutation = score.argsort(dim=1)[..., None, None, None].expand_as(x)
        sorted_x = x.gather(1, permutation)

        rank = torchsort.soft_rank(score, regularization_strength=self.sort_reg) - 1

        rank = rank[..., None, None, None].expand_as(x)
        left_idx = rank.long()
        right_idx = torch.remainder((left_idx + 1), x.shape[1])
        frac = rank.frac()
        left = torch.zeros_like(x).scatter_(0, left_idx, x).to(x.device)
        right = torch.zeros_like(x).scatter_(0, right_idx, x).to(x.device)
        soft_sorted_x = (1 - frac) * left + frac * right

        x = sorted_x + (soft_sorted_x - soft_sorted_x.detach())

        return x

    def unsort(self, x, score):
        permutation = score.argsort(dim=1)[..., None, None, None].expand_as(x)
        sorted_x = torch.zeros_like(x).scatter_(1, permutation, x).to(x.device)

        rank = torchsort.soft_rank(score) - 1

        rank = rank[..., None, None, None].expand_as(x)
        left_idx = rank.long()
        right_idx = torch.remainder((left_idx + 1), x.shape[1])
        frac = rank.frac()
        left = x.gather(1, left_idx)
        right = x.gather(1, right_idx)
        soft_sorted_x = (1 - frac) * left + frac * right

        x = sorted_x + (soft_sorted_x - soft_sorted_x.detach())

        return x


    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        predictions = self(inputs)

        loss = tmf.mean_absolute_error(predictions, targets) * 255

        metrics = {"train/loss": loss}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        predictions = self(inputs)

        loss = tmf.mean_absolute_error(predictions, targets) * 255

        if self.global_step == 0:
            wandb.define_metric("valid/loss", summary="min")

        metrics = {"valid/loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return predictions

    def on_after_backward(self):
        return super().on_after_backward()

    def configure_optimizers(self):
        # canonical_parameters = list(zip(*list(filter(lambda kv: "canonical_network" in kv[0], self.named_parameters()))))[1]
        # prediction_parameters = list(zip(*list(filter(lambda kv: "canonical_network" not in kv[0], self.named_parameters()))))[1]

        # canonical_optimizer = torch.optim.Adam(canonical_parameters, lr=self.learning_rate, weight_decay=1e-8)
        # prediction_optimizer = torch.optim.Adam(prediction_parameters, lr=self.learning_rate, weight_decay=1e-8)

        # # canonical_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(canonical_optimizer, self.patience)
        # canonical_scheduler = torch.optim.lr_scheduler.StepLR(canonical_optimizer, 100, gamma=0.4)
        # prediction_scheduler = torch.optim.lr_scheduler.StepLR(prediction_optimizer, 100, gamma=0.4)

        # return (
        #     [canonical_optimizer, prediction_optimizer],
        #     [
        #         {"scheduler": canonical_scheduler, "monitor": "valid/loss"},
        #         {"scheduler": prediction_scheduler, "monitor": "valid/loss"}
        #     ]
        # )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.patience, gamma=0.4)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/loss"}

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])

        self.logger.experiment.log(
            {
                "global_step": self.global_step,
            }
        )

