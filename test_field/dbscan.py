import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# === CONFIGURATION ===
input_path = r"D:\WORK\Python\Intern\Classification_task_1\test_field\air_mesh_2.ply"
output_path = os.path.splitext(input_path)[0] + "_geo_segmented.ply"

# === LOAD INPUT ===
mesh = o3d.io.read_triangle_mesh(input_path)
pcd = None

if mesh.is_empty() or not mesh.has_triangles():
    print("⚠️ Not a valid mesh. Loading as point cloud...")
    pcd = o3d.io.read_point_cloud(input_path)

# === FUNCTION TO CLASSIFY SHAPE TYPE ===
def classify_shape(points):
    if len(points) < 20:
        return "small"

    # PCA for geometric spread
    pca = PCA(n_components=3)
    pca.fit(points)
    ratios = pca.explained_variance_ratio_

    # Flat / planar surface → one strong direction
    if ratios[1] < 0.05:
        return "plane"

    # Cylindrical (2 dominant axes)
    elif ratios[2] < 0.05:
        return "cylinder"

    # Boxy or irregular
    else:
        return "complex"

# === COLOR MAP FOR CLASSES ===
shape_colors = {
    "plane": [0.2, 0.6, 1.0],     # Blue
    "cylinder": [1.0, 0.2, 0.2],  # Red
    "complex": [0.2, 1.0, 0.2],   # Green
    "small": [0.5, 0.5, 0.5],     # Gray
}

# === PROCESS ===
if pcd is None:
    print(f"✅ Loaded mesh with {len(mesh.triangles)} triangles, {len(mesh.vertices)} vertices")
    mesh.compute_vertex_normals()

    # Connected-component segmentation
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    vertex_colors = np.zeros_like(vertices)

    for cluster_id in np.unique(triangle_clusters):
        mask = triangle_clusters == cluster_id
        tri_points = vertices[triangles[mask].flatten()]

        shape_type = classify_shape(tri_points)
        color = shape_colors.get(shape_type, [0.8, 0.8, 0.8])

        vertex_colors[triangles[mask].flatten()] = color

        print(f"Segment {cluster_id}: {shape_type}, {len(tri_points)} pts")

    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"💾 Saved geometry-colored mesh to {output_path}")
    o3d.visualization.draw([mesh])

else:
    print(f"✅ Loaded point cloud with {len(pcd.points)} points")
    pcd.estimate_normals()

    # DBSCAN segmentation
    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=True))
    n_clusters = labels.max() + 1
    points = np.asarray(pcd.points)
    colors = np.zeros((len(points), 3))

    for cid in range(n_clusters):
        mask = labels == cid
        cluster_points = points[mask]
        shape_type = classify_shape(cluster_points)
        color = shape_colors.get(shape_type, [0.8, 0.8, 0.8])
        colors[mask] = color
        print(f"Cluster {cid}: {shape_type}, {len(cluster_points)} pts")

    # Assign and save
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"💾 Saved geometry-colored point cloud to {output_path}")
    o3d.visualization.draw([pcd])
