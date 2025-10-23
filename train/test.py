import torch
from torch.utils.data import DataLoader
from data_loader import PLYDataset, load_point_cloud
from nn import PointNetCls


class PointCloudTester:
    def __init__(self, model_path, classes, n_points=1024):
        self.model_path = model_path
        self.classes = classes
        self.n_points = n_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = PointNetCls(k=len(classes)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")

    def predict_file(self, ply_path):
        """Predict a single PLY file"""
        pc = load_point_cloud(ply_path, self.n_points)   # (N, 3)
        pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, N, 3)

        with torch.no_grad():
            outputs = self.model(pc)
            _, pred = torch.max(outputs, 1)

        predicted_class = self.classes[pred.item()]
        print(f"📂 {ply_path} → Predicted class: {predicted_class}")
        return predicted_class

    def evaluate_dataset(self, root, split="test", batch_size=16):
        """Evaluate accuracy on dataset"""
        dataset = PLYDataset(root, self.classes, split=split, n_points=self.n_points)
        loader = DataLoader(dataset, batch_size=batch_size)
        correct, total = 0, 0

        with torch.no_grad():
            for pcs, labels in loader:
                pcs, labels = pcs.to(self.device), labels.to(self.device)
                outputs = self.model(pcs)
                _, pred = torch.max(outputs, 1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100. * correct / total
        print(f"📊 {split} set accuracy: {acc:.2f}%")
        return acc


if __name__ == "__main__":
    classes = ["ifcDoor", "ifcWall", "ifcSlab", "ifcAirTerminal"]

    # Load model
    tester = PointCloudTester(model_path=r"D:\WORK\Python\Intern\Classification_task_1\train\pointnet_ifc_v9.pth", classes=classes, n_points=2048)

    # 🔹 Test on a single file
    tester.predict_file(r"D:\WORK\Python\Intern\Segmentation\test_field\ifcAirTernimal\test\4bc8923e82fe43edaf1cb5b2b609df43.ply")

    # 🔹 Or evaluate the whole test dataset 
    tester.evaluate_dataset(root="D:\WORK\Python\Intern\Segmentation\data\PLYdata", split="test")
