import torch.nn.functional as F
import kornia as K
from torch import nn
import torch
from canonical_network.models.equivariant_layers import RotationEquivariantConvLift, \
    RotoReflectionEquivariantConvLift, RotationEquivariantConv, RotoReflectionEquivariantConv
from torchvision import transforms
import numpy as np


class CanonizationNetwork(nn.Module):
    def __init__(self, in_shape, out_channels, kernel_size, group_type='rotation', num_rotations=4, num_layers=1, device='cuda'):
        super().__init__()
        if group_type == 'rotation':
            layer_list = [RotationEquivariantConvLift(in_shape[0], out_channels, kernel_size, num_rotations, device=device)]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())
                layer_list.append(RotationEquivariantConv(out_channels, out_channels, 1, num_rotations, device=device))
            self.eqv_network = nn.Sequential(*layer_list)
        elif group_type == 'roto-reflection':
            layer_list = [RotoReflectionEquivariantConvLift(in_shape[0], out_channels, kernel_size, num_rotations, device=device)]
            for i in range(num_layers - 1):
                layer_list.append(nn.ReLU())
                layer_list.append(RotoReflectionEquivariantConv(out_channels, out_channels, 1, num_rotations, device=device))
            self.eqv_network = nn.Sequential(*layer_list)
        else:
            raise ValueError('group_type must be rotation or roto-reflection for now.')
        out_shape = self.eqv_network(torch.zeros(1, *in_shape).to(device)).shape
        print('Canonization feature map shape:', out_shape)

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, group_size)
        """
        feature_map = self.eqv_network(x)
        feature_fibres = torch.mean(feature_map, dim=(1, 3, 4))
        return feature_fibres

class EquivariantCanonizationNetwork(nn.Module):
    def __init__(self,
                 base_encoder,
                 in_shape,
                 num_classes,
                 canonization_out_channels,
                 canonization_num_layers,
                 canonization_kernel_size,
                 canonization_beta = 1e4,
                 group_type='rotation',
                 num_rotations=4,
                 device='cuda',
                 batch_size=128):
        super().__init__()
        self.canonization_network = CanonizationNetwork(
            in_shape, canonization_out_channels, canonization_kernel_size,
            group_type, num_rotations, canonization_num_layers, device
        )
        print(self.canonization_network)
        self.base_encoder = base_encoder
        out_shape = self.base_encoder(torch.zeros(batch_size, *in_shape)).shape
        print('Base encoder feature map shape:', out_shape)
        print(self.base_encoder)
        if len(out_shape) == 4:
            self.predictor = nn.Linear(out_shape[1] * out_shape[2] * out_shape[3], num_classes)
        elif len(out_shape) == 2:
            self.predictor = nn.Linear(out_shape[1], num_classes)
        else:
            raise ValueError('Base encoder output shape must be 2 or 4 dimensional.')
        self.num_rotations = num_rotations
        self.group_type = group_type
        self.beta = canonization_beta
        self.num_group = num_rotations if group_type == 'rotation' else 2 * num_rotations

    def fibres_to_group(self, fibre_activations):
        device = fibre_activations.device
        #fibre_activations_one_hot = torch.nn.functional.softmax(self.beta * fibre_activations, dim=-1)
        fibre_activations_one_hot = torch.nn.functional.one_hot(torch.argmax(fibre_activations, dim=-1), self.num_group).float()
        fibre_activations_soft = torch.nn.functional.softmax(self.beta * fibre_activations, dim=-1)
        angles = torch.linspace(0., 360., self.num_rotations+1)[:self.num_rotations].to(device)
        angles = torch.cat([angles, angles], dim=0) if self.group_type == 'roto-reflection' else angles
        if self.training:
            #angle = torch.sum(fibre_activations_one_hot * angles, dim=-1)
            angle = torch.sum((fibre_activations_one_hot + fibre_activations_soft - fibre_activations_soft.detach()) * angles, dim=-1)
        else:
            angle = torch.sum(fibre_activations_one_hot * angles, dim=-1)
        if self.group_type == 'roto-reflection':
            reflect_one_hot = torch.cat(
                [torch.zeros(self.num_rotations), torch.ones(self.num_rotations)]
                , dim=0).to(device)
            if self.training:
                reflect_indicator = torch.sum((fibre_activations_one_hot + fibre_activations_soft - fibre_activations_soft.detach())
                                              * reflect_one_hot, dim=-1)
            else:
                reflect_indicator = torch.sum(fibre_activations_one_hot * reflect_one_hot, dim=-1)
            return angle, reflect_indicator
        else:
            return angle

    def inverse_action(self, x, fibres_activations):
        """
        x shape: (batch_size, in_channels, height, width)
        fibres_activations shape: (batch_size, group_size)
        :return: (batch_size, in_channels, height, width)
        """
        if self.group_type == 'rotation':
            angles = self.fibres_to_group(fibres_activations)
            group = [angles]
            x = K.geometry.rotate(x, -angles)
        elif self.group_type == 'roto-reflection':
            angles, reflect_indicator = self.fibres_to_group(fibres_activations)
            group = [angles, reflect_indicator]
            x_reflected = K.geometry.hflip(x)
            reflect_indicator = reflect_indicator[:,None,None,None]
            x = (1 - reflect_indicator) * x + reflect_indicator * x_reflected
            x = K.geometry.rotate(x, -angles)
        return x, group

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        :return: (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        x_canonized, group = self.get_canonized_images(x)
        reps = self.base_encoder(x_canonized)
        reps = reps.reshape(batch_size, -1)
        return self.predictor(reps)

    def get_canonized_images(self, x):
        fibres_activations = self.canonization_network(x)
        #x = transforms.Pad(4)(x)
        x_canonized, group = self.inverse_action(x, fibres_activations)
        return x_canonized, group


class VanillaNetwork(nn.Module):
    def __init__(self, encoder, in_shape, num_classes, batch_size=128, device='cuda'):
        super().__init__()
        self.encoder = encoder.to(device)
        print(self.encoder)
        out_shape = self.encoder(torch.zeros(batch_size, *in_shape).to(device)).shape
        print('feature map shape:', out_shape)

        if len(out_shape) == 4:
            self.predictor = nn.Linear(out_shape[1] * out_shape[2] * out_shape[3], num_classes)
        elif len(out_shape) == 2:
            self.predictor = nn.Linear(out_shape[1], num_classes)
        else:
            raise ValueError('Base encoder output shape must be 2 or 4 dimensional.')

    def forward(self, x):
        reps = self.encoder(x)
        reps = reps.view(x.shape[0], -1)
        return self.predictor(reps)

class PCACanonizationNetwork(nn.Module):
    def __init__(self, encoder, in_shape, num_classes, batch_size=128):
        super().__init__()
        self.encoder = encoder
        out_shape = self.encoder(torch.zeros(batch_size, *in_shape)).shape
        print('feature map shape:', out_shape)
        print(self.encoder)
        if len(out_shape) == 4:
            self.predictor = nn.Linear(out_shape[1] * out_shape[2] * out_shape[3], num_classes)
        elif len(out_shape) == 2:
            self.predictor = nn.Linear(out_shape[1], num_classes)
        else:
            raise ValueError('Base encoder output shape must be 2 or 4 dimensional.')

    def get_angles(self, images):
        images = images.reshape(images.shape[0], -1)
        device = images.device
        xs = np.linspace(-14, 14, num=28)
        ys = np.linspace(14, -14, num=28)
        x, y = np.meshgrid(xs, ys, )
        x, y = torch.tensor(x).float(), torch.tensor(y).float()
        angle_list = []
        for i in range(images.shape[0]):
            image = images[i]
            x_selected = x.flatten()[image > 0.5]
            y_selected = y.flatten()[image > 0.5]
            data = torch.cat([x_selected[:,None], y_selected[:,None]], dim=1)
            data = data - data.mean(dim=0)
            u, s, v = torch.svd(data)
            vect = v[:, torch.argmax(s)]
            angle_list.append(torch.atan2(vect[1], vect[0]) * 180 / np.pi)

        return torch.stack(angle_list).to(device)


    def get_canonized_images(self, x):
        angles = self.get_angles(x).detach()
        x_canonized = K.geometry.rotate(x, -angles)
        return x_canonized, angles

    def forward(self, x):
        batch_size = x.shape[0]
        x_canonized, angles = self.get_canonized_images(x)
        reps = self.encoder(x_canonized)
        reps = reps.view(batch_size, -1)
        return self.predictor(reps)


class BasicConvEncoder(nn.Module):
    def __init__(self, in_shape, out_channels, num_layers=6):
        super().__init__()
        encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(nn.Conv2d(in_shape[0], out_channels, 3, 1))
            elif i % 3 == 2:
                encoder_layers.append(nn.Conv2d(out_channels, 2 * out_channels, 5, 2, 1))
                out_channels *= 2
            else:
                encoder_layers.append(nn.Conv2d(out_channels, out_channels, 3, 1))
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU())
            if i % 3 == 2:
                encoder_layers.append(nn.Dropout2d(0.4))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x

class RotationEquivariantConvEncoder(nn.Module):
    def __init__(self, in_shape, out_channels, num_layers=6, num_rotations=4, device='cuda'):
        super().__init__()
        encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(RotationEquivariantConvLift(in_shape[0], out_channels, 3, num_rotations, 1, device=device))
            elif i % 3 == 2:
                encoder_layers.append(RotationEquivariantConv(out_channels, out_channels, 5, num_rotations, 2, 1, device=device))
                #out_channels *= 2
            else:
                encoder_layers.append(RotationEquivariantConv(out_channels, out_channels, 3, num_rotations, 1, device=device))

            encoder_layers.append(nn.BatchNorm3d(out_channels))
            encoder_layers.append(nn.ReLU())
            if i % 3 == 2:
                encoder_layers.append(nn.Dropout3d(0.4))

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x.mean(dim=2)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x