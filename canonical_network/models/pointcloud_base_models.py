import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
import torchmetrics.functional as tmf
import wandb

from canonical_network.models.vn_layers import *
from canonical_network.utils import to_categorical

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
SEGMENTATION_LABEL_TO_PART = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in SEGMENTATION_CLASSES.keys():
    for label in SEGMENTATION_CLASSES[cat]:
        SEGMENTATION_LABEL_TO_PART[label] = cat


class BasePointcloudModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super().__init__()
        self.num_parts = hyperparams.num_parts
        self.num_classes = hyperparams.num_classes
        self.train_rotation = hyperparams.train_rotation
        self.valid_rotation = hyperparams.valid_rotation
        self.num_points = hyperparams.num_points
        self.learning_rate = hyperparams.learning_rate if hasattr(hyperparams, "learning_rate") else None

        self.dummy_input = torch.zeros(2, 3, self.num_points, device=self.device, dtype=torch.float)
        self.dummy_indices = torch.zeros(2, 1, self.num_classes, device=self.device, dtype=torch.long)

        self.shape_ious = {cat: [] for cat in SEGMENTATION_CLASSES.keys()}

    def get_predictions(self, outputs):
        if type(outputs) == list:
            return torch.cat(outputs, dim=0)
        return outputs

    def training_step(self, batch, batch_idx):
        points, label, targets = batch
        points, label, targets = points.float(), label.long(), targets.long()

        if self.train_rotation == "z":
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True, device=self.device)
            points = trot.transform_points(points)
        elif self.train_rotation == "so3":
            trot = Rotate(R=random_rotations(points.shape[0]), device=self.device)
            points = trot.transform_points(points)

        points = points.transpose(2, 1)
        ont_hot_labels = to_categorical(label, self.num_classes).type_as(label)
        outputs = self(points, ont_hot_labels)
        predictions = self.get_predictions(outputs).squeeze()

        loss = self.get_loss(outputs, targets)
        accuracy = tmf.accuracy(predictions.permute(0, 2, 1), targets)

        metrics = {"train/loss": loss, "train/accuracy": accuracy}
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        points, label, targets = batch
        points, label, targets = points.float(), label.long(), targets.long()

        if self.valid_rotation == "z":
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True, device=self.device)
            points = trot.transform_points(points)
        elif self.valid_rotation == "so3":
            trot = Rotate(R=random_rotations(points.shape[0]), device=self.device)
            points = trot.transform_points(points)

        points = points.transpose(2, 1)
        ont_hot_labels = to_categorical(label, self.num_classes).type_as(label)
        outputs = self(points, ont_hot_labels)
        predictions = self.get_predictions(outputs)

        loss = self.get_loss(outputs, targets)
        accuracy = tmf.accuracy(predictions.permute(0, 2, 1), targets)
        self.update_cat_ious(points, predictions, targets)

        if self.global_step == 0:
            wandb.define_metric("valid/loss", summary="min")
            wandb.define_metric("valid/accuracy", summary="max")

        metrics = {"valid/loss": loss, "valid/accuracy": accuracy}
        self.log_dict(metrics, prog_bar=True)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, min_lr=1e-6, mode="max"
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid/mean_ious"}

    def validation_epoch_end(self, validation_step_outputs):
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.optimizer.param_groups[0]["lr"])

        if self.global_step == 0:
            wandb.define_metric("valid/mean_ious", summary="max")
        
        mean_ious = self.get_mean_ious()
        self.log_dict({"valid/mean_ious": mean_ious}, prog_bar=True)
        self.shape_ious = {cat: [] for cat in SEGMENTATION_CLASSES.keys()}

        predictions = self.get_predictions(validation_step_outputs)
        if self.current_epoch == 0:
            model_filename = f"canonical_network/results/shapenet/onnx_models/{self.model}_{wandb.run.name}_{str(self.global_step)}.onnx"
            if self.model == "pointnet":
                torch.onnx.export(self, (self.dummy_input.to(self.device), self.dummy_indices.to(self.device)), model_filename, opset_version=15)
            wandb.save(model_filename)

        self.logger.experiment.log(
            {
                "valid/logits": wandb.Histogram(predictions.to("cpu")),
                "global_step": self.global_step,
            }
        )

    def get_loss(self, outputs, targets):
        predictions = self.get_predictions(outputs).squeeze()
        return F.nll_loss(predictions.permute(0, 2, 1), targets)

    def update_cat_ious(self, points, predictions, targets):
        cur_batch_size, NUM_POINT, _ = points.size()

        cur_pred_val = predictions.cpu().data.numpy()
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        targets = targets.cpu().data.numpy()
        for i in range(cur_batch_size):
            cat = SEGMENTATION_LABEL_TO_PART[targets[i, 0]]
            logits = cur_pred_val_logits[i, :, :]
            cur_pred_val[i, :] = np.argmax(logits[:, SEGMENTATION_CLASSES[cat]], 1) + SEGMENTATION_CLASSES[cat][0]

        for i in range(cur_batch_size):
            segp = cur_pred_val[i, :]
            segl = targets[i, :]
            cat = SEGMENTATION_LABEL_TO_PART[segl[0]]
            part_ious = [0.0 for _ in range(len(SEGMENTATION_CLASSES[cat]))]
            for l in SEGMENTATION_CLASSES[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - SEGMENTATION_CLASSES[cat][0]] = 1.0
                else:
                    part_ious[l - SEGMENTATION_CLASSES[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l))
                    )
            self.shape_ious[cat].append(np.mean(part_ious))

    def get_mean_ious(self):
        all_shape_ious = []
        for cat in self.shape_ious.keys():
            for iou in self.shape_ious[cat]:
                all_shape_ious.append(iou)
            self.shape_ious[cat] = np.mean(self.shape_ious[cat])
        mean_shape_ious = np.mean(list(self.shape_ious.values()))
        return mean_shape_ious


class Pointnet(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(Pointnet, self).__init__(hyperparams)
        self.model = "pointnet"
        self.num_parts = hyperparams.num_parts
        self.normal_channel = hyperparams.normal_channel
        self.regularization_transform = hyperparams.regularization_transform

        if self.normal_channel:
            channel = 6
        else:
            channel = 3

        self.stn = STN3d(channel) if self.regularization_transform else None
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128) if self.regularization_transform else None
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, self.num_parts, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        if self.regularization_transform:
            trans = self.stn(point_cloud)
            point_cloud = point_cloud.transpose(2, 1)
            if D > 3:
                point_cloud, feature = point_cloud.split(3, dim=2)
            point_cloud = torch.bmm(point_cloud, trans)
            if D > 3:
                point_cloud = torch.cat([point_cloud, feature], dim=2)
            point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        if self.regularization_transform:
            trans_feat = self.fstn(out3)
            x = out3.transpose(2, 1)
            net_transformed = torch.bmm(x, trans_feat)
            net_transformed = net_transformed.transpose(2, 1)
        else:
            net_transformed = out3
            trans_feat = None

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048 + 16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.num_parts), dim=-1)
        net = net.view(B, N, self.num_parts)  # [B, N, 50]

        return net, trans_feat

    def get_predictions(self, outputs):
        if type(outputs) == list:
            outputs = list(zip(*outputs))
            return torch.cat(outputs[0], dim=0)
        return outputs[0]

    def get_loss(self, outputs, targets):
        predictions = outputs[0].permute(0, 2, 1)
        transformation_matrix = outputs[1]

        transformation_loss = (
            self.regularization_transform * self.feature_transform_reguliarzer(transformation_matrix)
            if self.regularization_transform
            else 0
        )
        total_loss = F.nll_loss(predictions, targets) + transformation_loss

        return total_loss

    def feature_transform_reguliarzer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss

class DGCNN(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(DGCNN, self).__init__(hyperparams)
        self.model = "DGCNN"
        self.num_parts = hyperparams.num_parts
        self.normal_channel = hyperparams.normal_channel
        self.regularization_transform = hyperparams.regularization_transform
        self.n_knn = hyperparams.n_knn
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.num_parts, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        B, D, N = x.size()
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1).type_as(x)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = x.transpose(1, 2).contiguous()

        net = F.log_softmax(x.view(-1, self.num_parts), dim=-1)
        net = net.view(B, N, self.num_parts)  # [B, N, 50]
        
        trans_feat = None
        return net, trans_feat


    def get_predictions(self, outputs):
        if type(outputs) == list:
            outputs = list(zip(*outputs))
            return torch.cat(outputs, dim=0)
        return outputs[0]


class VNPointnet(BasePointcloudModel):
    def __init__(self, hyperparams):
        super(VNPointnet, self).__init__(hyperparams)
        self.model = "vn_pointnet"
        self.n_knn = hyperparams.n_knn
        self.normal_channel = hyperparams.normal_channel
        self.num_parts = hyperparams.num_parts
        self.pooling = hyperparams.pooling
        if self.normal_channel:
            channel = 6
        else:
            channel = 3

        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 512 // 3, dim=4, negative_slope=0.0)

        self.conv5 = VNLinear(512 // 3, 2048 // 3)
        self.bn5 = VNBatchNorm(2048 // 3, dim=4)

        self.std_feature = VNStdFeature(2048 // 3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)

        if self.pooling == "max":
            self.pool = VNMaxPool(64 // 3)
        elif self.pooling == "mean":
            self.pool = mean_pool

        self.fstn = VNSTNkd(hyperparams, d=128 // 3)

        self.convs1 = torch.nn.Conv1d(9025, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, self.num_parts, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.n_knn)
        point_cloud = self.conv_pos(feat)
        point_cloud = self.pool(point_cloud)

        out1 = self.conv1(point_cloud)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        net_global = self.fstn(out3).unsqueeze(-1).repeat(1, 1, 1, N)
        net_transformed = torch.cat((out3, net_global), 1)

        out4 = self.conv4(net_transformed)
        out5 = self.bn5(self.conv5(out4))

        out5_mean = out5.mean(dim=-1, keepdim=True).expand(out5.size())
        out5 = torch.cat((out5, out5_mean), 1)
        out5, trans = self.std_feature(out5)
        out5 = out5.view(B, -1, N)

        out_max = torch.max(out5, -1, keepdim=False)[0]

        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        expand = out_max.view(-1, 2048 // 3 * 6 + 16, 1).repeat(1, 1, N)

        out1234 = torch.cat((out1, out2, out3, out4), dim=1)
        out1234 = torch.einsum("bijm,bjkm->bikm", out1234, trans).view(B, -1, N)

        concat = torch.cat([expand, out1234, out5], 1)

        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.num_parts), dim=-1)
        net = net.view(B, N, self.num_parts)  # [B, N, 50]

        return net


class VNPointnetSmall(pl.LightningModule):
    def __init__(self, hyperparams):
        super(VNPointnetSmall, self).__init__()
        self.model = "vn_pointnet"
        self.n_knn = hyperparams.n_knn
        self.normal_channel = hyperparams.normal_channel
        self.num_parts = hyperparams.num_parts
        self.num_classes = hyperparams.num_classes
        self.pooling = hyperparams.pooling
        if self.normal_channel:
            channel = 6
        else:
            channel = 3

        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU(128 // 3 * 2, 256 // 3, dim=4, negative_slope=0.0)

        self.conv5 = VNLinearLeakyReLU(256 // 3, 256 // 3, dim=4, negative_slope=0.0)
        self.bn5 = VNBatchNorm(256 // 3, dim=4)

        self.conv6 = VNBilinear(256 // 3, self.num_classes, 12 // 3)

        if self.pooling == "max":
            self.pool = VNMaxPool(64 // 3)
        elif self.pooling == "mean":
            self.pool = mean_pool

        self.fstn = VNSTNkd(hyperparams, d=128 // 3)

    def forward(self, point_cloud, labels):
        B, D, N = point_cloud.size()

        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.n_knn)
        point_cloud = self.conv_pos(feat)
        point_cloud = self.pool(point_cloud)

        out1 = self.conv1(point_cloud)
        out2 = self.conv2(out1)

        net_global = self.fstn(out2).unsqueeze(-1).repeat(1, 1, 1, N)
        net_transformed = torch.cat((out2, net_global), 1)

        out4 = self.conv4(net_transformed)
        out5 = self.bn5(self.conv5(out4))

        out5_mean = out5.mean(dim=-1, keepdim=False)

        out = self.conv6(out5_mean, labels)

        return out


class STN3d(pl.LightningModule):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)))
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(pl.LightningModule):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class VNSTNkd(pl.LightningModule):
    def __init__(self, hyperparams, d):
        super(VNSTNkd, self).__init__()
        self.conv1 = VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0)

        if hyperparams.pooling == "max":
            self.pool = VNMaxPool(1024 // 3)
        elif hyperparams.pooling == "mean":
            self.pool = mean_pool

        self.fc3 = VNLinear(256 // 3, d)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


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
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    idx_base = torch.arange(0, batch_size).type_as(idx).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
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
