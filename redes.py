# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f


class MnistNet(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=30, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * 5 * 5, 100)  # Adjust the input features
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_outputs)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax to the output

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Inicialización He para capas convolucionales
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Inicialización uniforme para capas lineales
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # a es el factor de la rectificación lineal
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)


class DigitFiveNet(nn.Module):
    """Monta la cnn"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding="same")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding="same")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding="same")
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.pool1(f.relu(self.conv1(x)))
        x2 = self.pool2(f.relu(self.conv2(x1)))
        x3 = self.pool3(f.relu(self.conv3(x2)))
        x4 = self.pool4(f.relu(self.conv4(x3)))

        x = torch.flatten(x4, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TurtlebotNet(nn.Module):
    def __init__(self) -> None:
        super(TurtlebotNet, self).__init__()
        self.layer1 = nn.Linear(1440, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x
