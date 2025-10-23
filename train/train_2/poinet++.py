import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Utilities ----
def square_distance(src, dst):
    # src: (B, N, C) dst: (B, M, C)
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)) ** 2, dim=-1)

def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def knn(x, k):
    # x: (B, N, C)
    dist = square_distance(x, x)  # (B, N, N)
    idx = dist.argsort()[:, :, 1:k+1]  # exclude self
    return idx

# ---- PointNet Set Abstraction ----
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, k, mlp):
        super().__init__()
        self.npoint = npoint
        self.k = k
        layers = []
        last_channel = 3
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz):
        # xyz: (B, N, 3)
        B, N, C = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)               # (B, npoint, 3)

        # group neighbors
        idx = knn(new_xyz, self.k)                         # (B, npoint, k)
        grouped_xyz = index_points(xyz, idx)               # (B, npoint, k, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)   # local coords

        # (B, 3, npoint, k)
        grouped_xyz = grouped_xyz.permute(0, 3, 1, 2)
        new_points = self.mlp(grouped_xyz)                 # (B, mlp[-1], npoint, k)
        new_points = torch.max(new_points, -1)[0]          # (B, mlp[-1], npoint)
        return new_xyz, new_points

# ---- PointNet++ Classification ----
class PointNet2Cls(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, k=32, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, k=64, mlp=[128, 128, 256])
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x: (B, N, 3)
        _, x = self.sa1(x)
        _, x = self.sa2(x)
        x = torch.max(x, 2)[0]        # (B, 256)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
