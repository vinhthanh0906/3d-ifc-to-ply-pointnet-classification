import os
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from plyfile import PlyData, PlyElement

# Input and output dataset folders
input_root = r"D:\WORK\Python\Intern\Segmentation\data\IFCNetCore.IFC"
output_root = r"D:\WORK\Python\Intern\Segmentation\data\PLYNetCore.PLY"

def ifc_to_ply(ifc_path, ply_path, target_class=None):
    try:
        ifc_file = ifcopenshell.open(ifc_path)
        schema = ifc_file.schema
        print(f"📂 Processing {ifc_path} (Schema: {schema}, Class: {target_class})")

        settings = ifcopenshell.geom.settings()
        verts_all, faces_all = [], []
        vertex_offset = 0

        # Try target class first (if given)
        elements = ifc_file.by_type(target_class) if target_class else []

        # Fallback to all products if target not found
        if not elements:
            if target_class:
                print(f"⚠️ No {target_class} found in {ifc_path}, using IfcProduct instead")
            elements = ifc_file.by_type("IfcProduct")

        for element in elements:
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
                verts = shape.geometry.verts
                faces = shape.geometry.faces

                grouped_verts = [[verts[i], verts[i+1], verts[i+2]]
                                 for i in range(0, len(verts), 3)]
                grouped_faces = [[faces[i] + vertex_offset,
                                  faces[i+1] + vertex_offset,
                                  faces[i+2] + vertex_offset]
                                 for i in range(0, len(faces), 3)]

                verts_all.extend(grouped_verts)
                faces_all.extend(grouped_faces)
                vertex_offset += len(grouped_verts)

            except Exception:
                continue  # skip element if no geometry

        if not verts_all:
            print(f"⚠️ No geometry extracted from {ifc_path}")
            return

        # Convert to numpy structured arrays
        vertex = np.array([(x, y, z) for x, y, z in verts_all],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        face = np.array([(f,) for f in faces_all],
                        dtype=[('vertex_indices', 'i4', (3,))])

        # Create PLY
        vertex_element = PlyElement.describe(vertex, 'vertex')
        face_element = PlyElement.describe(face, 'face')
        ply_data = PlyData([vertex_element, face_element])

        # Ensure output folder exists
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)

        ply_data.write(ply_path)
        print(f"✅ Converted -> {ply_path}")

    except Exception as e:
        print(f"❌ Error converting {ifc_path}: {e}")


# Walk through dataset
for root, dirs, files in os.walk(input_root):
    for filename in files:
        if filename.lower().endswith(".ifc"):
            ifc_path = os.path.join(root, filename)

            # Mirror folder structure in output
            relative_path = os.path.relpath(ifc_path, input_root)
            ply_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + ".ply")

            # Guess IFC class from top-level class folder (e.g., "IfcSlab")
            class_name = os.path.normpath(relative_path).split(os.sep)[0]
            target_class = class_name if class_name.lower().startswith("ifc") else None

            ifc_to_ply(ifc_path, ply_path, target_class=target_class)
