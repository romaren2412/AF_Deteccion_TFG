import torch
import torch.nn as nn
import torch.nn.functional as f


class MnistNetFLTrust(nn.Module):
    def __init__(self):
        super(MnistNetFLTrust, self).__init__()
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


class MnistNetFLARE(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(MnistNetFLARE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=30, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # La salida de la segunda capa de max pooling sigue siendo 5x5x5
        self.fc1 = nn.Linear(5 * 5 * 5, 100)  # Primera capa completamente conectada
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_outputs)  # Capa antes de la salida final

        self.initialize_weights()

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return f.softmax(x, dim=1)  # Usamos softmax para la salida final

    def extraer_plr(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        plr = self.fc2(x)
        return plr

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
