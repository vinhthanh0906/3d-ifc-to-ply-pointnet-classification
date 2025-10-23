# nn.py
import torch
from torch import nn
from torch.nn import functional as F


class PointNetCls(nn.Module):
    """
    Backwards-compatible PointNet classifier.

    You can initialize it either with:
        PointNetCls(num_classes=14)     # preferred
    or (legacy):
        PointNetCls(k=14)

    If neither is provided, default = 14.
    """
    def __init__(self, num_classes=None, k=None):
        super(PointNetCls, self).__init__()

        # Determine number of output classes (backwards-compatible)
        if num_classes is None and k is None:
            num_classes = 14
        elif num_classes is None:
            num_classes = k
        # else num_classes already provided

        self.num_classes = int(num_classes)

        # simple MLP PointNet-like architecture (same shape as you gave)
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        """
        x: (B, N, 3) tensor
        returns: (B, num_classes)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # global max-pooling along point dimension (dim=1)
        x = torch.max(x, 1)[0]
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        out = self.fc6(x)
        return out
