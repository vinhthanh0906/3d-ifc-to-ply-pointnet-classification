import os
import open3d as o3d
import numpy as np

# === CONFIG ===
input_folder = r"D:\WORK\Python\Intern\Segmentation\data\PLYdata\ifcAirTerminal\train"   # folder containing .ply files
output_folder = r"D:\WORK\Python\Intern\Segmentation\test_field\ifcAirTernimal\train"       # folder to save augmented .ply files
extra_points = 2048            # number of extra points to add with Poisson-disk

os.makedirs(output_folder, exist_ok=True)

def augment_pointcloud(file_path, out_path, extra_points):
    # Load original point cloud
    pcd = o3d.io.read_point_cloud(file_path)

    # Convert to mesh to use Poisson-disk sampling
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound

    # Create dense random points in bbox
    random_pts = np.random.uniform(low=min_bound, high=max_bound, size=(extra_points*5, 3))
    random_pcd = o3d.geometry.PointCloud()
    random_pcd.points = o3d.utility.Vector3dVector(random_pts)

    # Apply Poisson-disk sampling to get evenly distributed points
    sampled_pcd = random_pcd.farthest_point_down_sample(extra_points)

    # Merge with original
    merged_points = np.vstack((np.asarray(pcd.points), np.asarray(sampled_pcd.points)))
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

    # Save output
    o3d.io.write_point_cloud(out_path, merged_pcd)
    print(f"Saved augmented point cloud: {out_path}")

# Process all PLY files
for fname in os.listdir(input_folder):
    if fname.endswith(".ply"):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        augment_pointcloud(in_path, out_path, extra_points)

print("✅ Finished augmenting all point clouds.")
