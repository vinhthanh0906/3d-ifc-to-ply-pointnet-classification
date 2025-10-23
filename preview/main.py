'''
D:\WORK\Python\Intern\Segmentation\data\IFCNetCore.IFC\IfcAirTerminal\test\0b39c7a0e3fb421dbdc1d2d0a55e5b72.ifc
'''
import ifcopenshell
import ifcopenshell.geom
import matplotlib.pyplot as plt
import os

# Đường dẫn file IFC
ifc_file = ifcopenshell.open(r"D:\WORK\Python\Intern\Segmentation\data\IFCNetCore.IFC\IfcAirTerminal\test\0b39c7a0e3fb421dbdc1d2d0a55e5b72.ifc")

# Output folder
out_dir = "dataset/images"
os.makedirs(out_dir, exist_ok=True)

# IFC categories bạn quan tâm
classes = ["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow"]

# Lặp qua từng class
for cls in classes:
    elements = ifc_file.by_type(cls)
    for i, elem in enumerate(elements[:50]):  # ví dụ lấy 50 mẫu đầu
        shape = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), elem)
        verts = shape.geometry.verts
        faces = shape.geometry.faces

        # Simple render (matplotlib 3D)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        verts_np = np.array(verts).reshape(-1, 3)
        faces_np = np.array(faces).reshape(-1, 3)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts_np[faces_np], alpha=0.8)
        ax.add_collection3d(mesh)
        ax.set_axis_off()
        plt.savefig(f"{out_dir}/{cls}_{i}.png")
        plt.close()
