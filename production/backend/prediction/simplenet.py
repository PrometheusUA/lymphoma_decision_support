import torch
import torch.nn.functional as F
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 7, padding="same")
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(20, 50, 5, padding="same")
        self.conv3 = nn.Conv2d(50, 120, 3, padding="same")
        self.fc1 = nn.Linear(120 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = self.pool(F.tanh(self.conv3(x)))
        # x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
