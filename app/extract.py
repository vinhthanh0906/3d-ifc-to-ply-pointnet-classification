import os

dataset_path = r"D:\WORK\Python\Intern\Classification_task_1\data\PLYNetCore.PLY"  # change this

# Get all folders (classes)
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

print("Found classes:")
for c in classes:
    print("-", c)
