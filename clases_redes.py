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
