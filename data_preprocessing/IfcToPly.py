import os
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from plyfile import PlyData, PlyElement

# Input and output folders
input_folder = r"D:\WORK\Python\Intern\Segmentation\data\IFCMini\IfcSlab\train"
output_folder = r"D:\WORK\Python\Intern\Segmentation\data\PLYdata\ifcSlab\train"

os.makedirs(output_folder, exist_ok=True)

def ifc_to_ply(ifc_path, ply_path, target_class="IfcSlab"):
    try:
        ifc_file = ifcopenshell.open(ifc_path)
        schema = ifc_file.schema  # e.g., "IFC2X3" or "IFC4"
        print(f"📂 Processing {ifc_path} (Schema: {schema})")

        settings = ifcopenshell.geom.settings()
        verts_all, faces_all = [], []
        vertex_offset = 0

        # First, try target class (IfcAirTerminal)
        elements = ifc_file.by_type(target_class)

        # If nothing found, fallback to all products
        if not elements:
            print(f"⚠️ No {target_class} found in {ifc_path}, using IfcProduct instead")
            elements = ifc_file.by_type("IfcProduct")

        for element in elements:
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
                verts = shape.geometry.verts
                faces = shape.geometry.faces

                # Group into triplets
                grouped_verts = [[verts[i], verts[i+1], verts[i+2]] 
                                 for i in range(0, len(verts), 3)]
                grouped_faces = [[faces[i]+vertex_offset,
                                  faces[i+1]+vertex_offset,
                                  faces[i+2]+vertex_offset]
                                 for i in range(0, len(faces), 3)]

                verts_all.extend(grouped_verts)
                faces_all.extend(grouped_faces)
                vertex_offset += len(grouped_verts)

            except Exception:
                continue  # skip if element has no geometry

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

        ply_data.write(ply_path)
        print(f"✅ Converted -> {ply_path}")

    except Exception as e:
        print(f"❌ Error converting {ifc_path}: {e}")


# Loop over all IFC files in folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".ifc"):
        ifc_path = os.path.join(input_folder, filename)
        ply_filename = os.path.splitext(filename)[0] + ".ply"
        ply_path = os.path.join(output_folder, ply_filename)
        ifc_to_ply(ifc_path, ply_path)
