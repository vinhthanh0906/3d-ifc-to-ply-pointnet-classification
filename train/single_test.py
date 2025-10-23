import torch
from torch.utils.data import DataLoader
from data_loader import PLYDataset
from nn import PointNetCls


class SingleClassTester:
    def __init__(self, model_path, classes, n_points=1024):
        self.classes = classes
        self.n_points = n_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = PointNetCls(k=len(classes)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")

    def evaluate_single_class(self, root, class_name, split="test", batch_size=16):
        # Dataset root should be the dataset folder, not inside the class
        dataset = PLYDataset(root, [class_name], split=split, n_points=self.n_points)
        loader = DataLoader(dataset, batch_size=batch_size)

        if len(dataset) == 0:
            print(f"⚠️ No data found for class {class_name} in {root}/{class_name}/{split}")
            return 0.0

        correct, total = 0, 0
        with torch.no_grad():
            for pcs, labels in loader:
                pcs, labels = pcs.to(self.device), labels.to(self.device)
                outputs = self.model(pcs)
                _, pred = torch.max(outputs, 1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100. * correct / total
        print(f"📊 Accuracy for class '{class_name}' ({split} set): {acc:.2f}%")
        return acc


if __name__ == "__main__":
    # Define classes (same order as in training)
    classes = ["ifcAirTerminal", "ifcDoor", "ifcSlab", "ifcWall"]

    # Initialize tester with model
    tester = SingleClassTester(
        model_path="D:\WORK\Python\Intern\Segmentation\models\pointnet_ifc_v6.pth",
        classes=classes,
        n_points=2048
    )

    # Run test for a single class
    dataset_root = r"D:\WORK\Python\Intern\Segmentation\data\pointcloud_data"
    tester.evaluate_single_class(root=dataset_root, class_name="ifcAirTerminal", split="test")
