# coding: utf-8
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

        self.drop = nn.Dropout(p=0.2)
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

    def extraer_plr(self, x):
        x1 = self.pool1(f.relu(self.conv1(x)))
        x2 = self.pool2(f.relu(self.conv2(x1)))
        x3 = self.pool3(f.relu(self.conv3(x2)))
        x4 = self.pool4(f.relu(self.conv4(x3)))

        x = torch.flatten(x4, 1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        plr = self.fc4(x)
        return plr

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

    def extraer_plr(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        plr = self.layer5(x)
        return plr
