import torch.nn as nn


class MLR(nn.Module):
    def __init__(self, input_size, num_classes):
        """
            Inicializa la clase MLR.
            Parámetros:
                - input_size: Tamaño de la entrada, número de características en cada ejemplo.
                - num_classes: Número de clases en el problema de clasificación.
        """
        super(MLR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


class VAE(nn.Module):
    def __init__(self, grad_len):
        super(VAE, self).__init__()
        # 2 capas: encoder y decoder
        self.encoder = nn.sequential(
            nn.Linear(grad_len, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, grad_len)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TwoLayerFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(TwoLayerFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=30, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=50, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(50 * 4 * 4, 512)  # Assuming input size after flattening is 4x4
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_outputs)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
