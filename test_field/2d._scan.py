import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

mesh = o3d.io.read_triangle_mesh(r"D:\WORK\Python\Intern\Classification_task_1\test_field\air_mesh_2.ply")
mesh.compute_vertex_normals()

# Convert to point cloud for easier projection
pcd = mesh.sample_points_uniformly(number_of_points=200000)
points = np.asarray(pcd.points)

# Project onto Y-Z plane (side view)
y, z = points[:, 1], points[:, 2]

plt.figure(figsize=(8, 8))
plt.scatter(y, z, s=0.2, c='k')
plt.xlabel("Y axis (width)")
plt.ylabel("Z axis (height)")
plt.title("2D Side View Projection (Y-Z plane)")
plt.axis("equal")
plt.grid(False)
plt.tight_layout()
plt.savefig("air_mesh_sideview.png", dpi=600)
plt.show()
