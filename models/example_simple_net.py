import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseNet


"""
An example of child class of the BaseNet for
the CIFAR10 classification example.
Architecture taken from PyTorch tutorial.
"""


class ExampleSimpleNet(BaseNet):
    def __init__(self):
        super(ExampleSimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            return torch.argmax(
                self.forward(x),
                axis=1)
