import torch
import torch.nn as nn
import torch.nn.functional as f


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3)
        self.fc1 = nn.Linear(50 * 5 * 5, 100)  # Tamaño ajustado según el output de la última capa Max Pooling
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)

    def extraer_plr(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        plr = self.fc2(x)
        return plr
