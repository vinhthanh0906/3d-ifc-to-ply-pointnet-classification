import os

# === CONFIG ===
dataset_root = r"D:\WORK\Python\Intern\Segmentation_task_2\data\PLYNetCore.FULL"   # root dataset folder

def rename_dataset(dataset_root):
    for class_name in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_path):
            continue  # skip files

        for split in ["train", "test"]:
            split_path = os.path.join(class_path, split)
            if not os.path.exists(split_path):
                continue

            # List only .ply files
            ply_files = [f for f in os.listdir(split_path) if f.endswith(".ifc")]
            ply_files.sort()  # keep order consistent

            for idx, fname in enumerate(ply_files, start=1):
                old_path = os.path.join(split_path, fname)
                new_name = f"{class_name}_{split}_{idx:04d}.ply"
                new_path = os.path.join(split_path, new_name)

                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} → {new_path}")

    print("✅ Finished renaming all .ifc files.")

# Run
rename_dataset(dataset_root)
