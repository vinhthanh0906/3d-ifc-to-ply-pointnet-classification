import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import PLYDataset, load_point_cloud
from nn import PointNetCls


class PointCloudTrainer:
    def __init__(
        self,
        root,
        n_points=2048,
        batch_size=16,
        lr=0.001,
        epochs=100,
        save_path="pointnet_model.pth",
    ):
        self.root = root
        self.n_points = n_points
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.save_path = save_path

        # ✅ Auto-detect classes (folders in root)
        self.classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        print(f"📁 Detected classes: {self.classes}")

        # ✅ Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")

        # ✅ Create datasets & loaders
        self.train_dataset = PLYDataset(root, self.classes, split="train", n_points=n_points)
        self.test_dataset = PLYDataset(root, self.classes, split="test", n_points=n_points)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # ✅ Model setup
        self.model = PointNetCls(k=len(self.classes)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    # ===============================
    # 🔹 Train Loop
    # ===============================
    def train(self):
        best_acc = 0.0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
            for pcs, labels in pbar:
                pcs, labels = pcs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(pcs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_acc = 100.0 * correct / total
            avg_loss = total_loss / len(self.train_loader)
            print(f"📊 Epoch [{epoch}/{self.epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

            # Evaluate every epoch
            val_acc = self.evaluate()
            if val_acc > best_acc:
                best_acc = val_acc
                self.save("best_model.pth")

        print("✅ Training complete. Best validation accuracy:", f"{best_acc:.2f}%")

    # ===============================
    # 🔹 Evaluation
    # ===============================
    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for pcs, labels in self.test_loader:
                pcs, labels = pcs.to(self.device), labels.to(self.device)
                outputs = self.model(pcs)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total
        print(f"🔍 Validation Accuracy: {acc:.2f}%")
        return acc

    # ===============================
    # 🔹 Save / Load
    # ===============================
    def save(self, path=None):
        if path is None:
            path = self.save_path
        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = self.save_path
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"📂 Model loaded from {path}")


# ===============================
# 🔹 Inference Class
# ===============================
class PointCloudClassifier:
    def __init__(self, model, classes, n_points=2048):
        self.model = model
        self.classes = classes
        self.n_points = n_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, ply_path):
        pc = load_point_cloud(ply_path, self.n_points)
        pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(pc)
            _, pred = outputs.max(1)

        label = self.classes[pred.item()]
        print(f"🎯 Predicted class: {label}")
        return label


# ===============================
# 🔹 Run Training + Example
# ===============================
if __name__ == "__main__":
    root = r"D:\WORK\Python\Intern\Classification_task_1\data\PLYNetCore.PLY"

    trainer = PointCloudTrainer(root, epochs=100, batch_size=16, lr=0.001)
    trainer.train()

    # Evaluate final model
    trainer.evaluate()
    trainer.save()

    # Example inference
    classifier = PointCloudClassifier(trainer.model, trainer.classes)
    test_file = r"D:\WORK\Python\Intern\Segmentation\data\PLYdata\ifcWall\test\0acdcd38b2e84cfaa8eafed1a8ef4111.ply"
    classifier.predict(test_file)
