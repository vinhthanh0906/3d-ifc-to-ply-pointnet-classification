import torch
from torch import nn
from torch.nn import functional as F


class PointNetCls(nn.Module):
    def __init__(self, num_classes=14):
        """
        Simple PointNet-style classifier.
        Args:
            num_classes (int): Number of output classes.
        """
        super(PointNetCls, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)

    def forward(self, x):  # x: (B, N, 3)
        """
        Forward pass through the model.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.max(x, 1)[0]  # Global max pooling over N points
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        out = self.fc6(x)
        return out
