import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData
from nn import PointNetCls  # your model definition file
import os

# ---- Config ----
MODEL_PATH = "D:\WORK\Python\Intern\Classification_task_1\models\pointnet_model.pth"  # path to trained model
PLY_PATH = r"D:\WORK\Python\Intern\Segmentation_task_2\test_field\valve.plyy"        # path to the PLY file
NUM_POINTS = 2048              # same as training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- IFC class labels ----
CLASS_NAMES = [
    "IfcAirTerminal",
    "IfcBeam",
    "IfcCableCarrierFitting",
    "IfcCableCarrierSegment",
    "IfcDoor",
    "IfcDuctFitting",
    "IfcDuctSegment",
    "IfcFurniture",
    "IfcOutlet",
    "IfcPipeFitting",
    "IfcPlate",
    "IfcRailing",
    "IfcSanitaryTerminal",
    "IfcSlab",
    "IfcSpaceHeater",
    "IfcStair",
    "IfcValve",
    "IfcWall"
]

# ---- Load model ----
model = PointNetCls(k=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---- Function to load vertices from PLY ----
def load_ply_vertices(ply_path, npoints=2048):
    plydata = PlyData.read(ply_path)
    vertices = np.stack([
        plydata['vertex'].data['x'],
        plydata['vertex'].data['y'],
        plydata['vertex'].data['z']
    ], axis=-1)
    if len(vertices) >= npoints:
        idx = np.random.choice(len(vertices), npoints, replace=False)
    else:
        idx = np.random.choice(len(vertices), npoints, replace=True)
    return vertices[idx, :3].astype(np.float32)

# ---- Predict single PLY ----
points = load_ply_vertices(PLY_PATH, NUM_POINTS)
points = torch.from_numpy(points).unsqueeze(0).to(DEVICE)  # (1, N, 3)

with torch.no_grad():
    logits = model(points)
    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_class = CLASS_NAMES[pred_idx]

print(f"Predicted class: {pred_class}")
print(f"Class probabilities: {probs.squeeze().cpu().numpy()}")
