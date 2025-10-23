import numpy as np 
from plyfile import PlyData
from torch.utils.data import Dataset
import os 
import torch


def load_point_cloud(ply_path, n_points=2024):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    # Normalize to unit sphere
    points = points - np.mean(points, axis=0)
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale

    # Random sample
    if len(points) >= n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
    else:
        idx = np.random.choice(len(points), n_points, replace=True)
    return points[idx]


class PLYDataset(Dataset):
    def __init__(self, root_dir, classes, split="train", n_points=1024):
        self.samples = []
        self.classes = classes
        self.n_points = n_points

        for i, cls in enumerate(classes):
            folder = os.path.join(root_dir, cls, split)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith(".ply"):
                    self.samples.append((os.path.join(folder, f), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ply_path, label = self.samples[idx]
        pc = load_point_cloud(ply_path, self.n_points)
        return torch.tensor(pc, dtype=torch.float32), label