import torch.nn as nn
import torch.nn.functional as F


class CNN_v1(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(CNN_v1, self).__init__()
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


class CNN_v2(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=30, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Calculate the size of the output from the last max pooling layer
        # After first conv and pool: (28 - 3) + 1 = 26, then 26 / 2 = 13 (because of max pooling)
        # After second conv and pool: (13 - 3) + 1 = 11, then 11 / 2 = 5.5 which is rounded down to 5
        # The output size is 5x5x5
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
