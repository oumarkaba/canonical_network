#%%
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms

from kornia.geometry.transform import rotate
import matplotlib.pyplot as plt



mnist = datasets.MNIST('.', train=True, download=True)
image = transforms.ToTensor()(mnist[0][0]).unsqueeze(0)

target_image = rotate(image, torch.FloatTensor([90]))


class EnergyFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.ReLU(),
        )
        self.lin = nn.Linear(16 * 7 * 7, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.lin(x.flatten(1))
        return x
    



def min_energy(input):
    # rot = torch.rand(1).requires_grad_(True) * 360
    rot = torch.zeros(1).requires_grad_(True)
    for _ in range(5):
        rotated = rotate(input, rot)
        energy = energy_function(rotated)
        g, = torch.autograd.grad(energy, rot, only_inputs=True, create_graph=True)
        # print(g)
        rot = rot - 10000*g
    return rotate(image, rot)


energy_function = EnergyFunction()
opt = torch.optim.Adam(energy_function.parameters(), 1e-1)
for i in range(100):
    canonical = min_energy(image)
    loss = nn.functional.mse_loss(canonical, target_image)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(loss)
    # plt.imshow(canonical.detach().squeeze())
    plt.show()

# print('target')
# plt.imshow(target_image.detach().squeeze())
# plt.show()

