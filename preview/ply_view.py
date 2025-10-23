import open3d as o3d
import numpy as np

def visualize_pointcloud(ply_path, n_points=2048):
    # Load as point cloud
    pcd = o3d.io.read_point_cloud(ply_path)

    if len(pcd.points) == 0:
        print("❌ No points found in this PLY file")
        return

    # Convert to numpy
    pts = np.asarray(pcd.points)

    # If too many points, randomly sample
    if len(pts) > n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    elif len(pts) < n_points:
        # If too few points, duplicate randomly
        idx = np.random.choice(len(pts), n_points, replace=True)
        pts = pts[idx]

    # Convert back to Open3D point cloud
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(pts)

    # Visualize
    o3d.visualization.draw_geometries([sampled_pcd])


# Example
ply_file = r"D:\WORK\Python\Intern\Segmentation\test_field\wall_1.ply"
visualize_pointcloud(ply_file, n_points=2048)
