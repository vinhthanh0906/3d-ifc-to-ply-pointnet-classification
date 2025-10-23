import ifcopenshell
import ifcopenshell.geom
from OCC.Display.SimpleGui import init_display
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

# -------------------------------------------------
# 1. Load IFC file
# -------------------------------------------------
ifc_path = r"D:\WORK\Python\Intern\Segmentation\data\IFCNetCore.IFC\IfcSlab\test\0d2300c7d58646079d35ca7136e0a38d.ifc"  # change path
ifc_file = ifcopenshell.open(ifc_path)

# -------------------------------------------------
# 2. Find all beams
# -------------------------------------------------
beams = ifc_file.by_type("IfcSlab")

if not beams:
    print("⚠️ No beams found in this IFC file.")
else:
    print(f"✅ Found {len(beams)} beam(s):")
    for beam in beams:
        print(f" - GlobalId: {beam.GlobalId}, Name: {beam.Name}")

# -------------------------------------------------
# 3. Set up 3D Viewer
# -------------------------------------------------
display, start_display, add_menu, add_function_to_menu = init_display()
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_PYTHON_OPENCASCADE, True)

# -------------------------------------------------
# 4. Display beams (in red)
# -------------------------------------------------
for beam in beams:
    try:
        shape = ifcopenshell.geom.create_shape(settings, beam).geometry
        ais_shape = AIS_Shape(shape)
        ais_shape.SetColor(Quantity_Color(0.9, 0.2, 0.2, Quantity_TOC_RGB))  # red color
        display.Context.Display(ais_shape, True)
    except Exception as e:
        print(f"⚠️ Could not render beam {beam.GlobalId}: {e}")

# -------------------------------------------------
# 5. Start Viewer
# -------------------------------------------------
display.FitAll()
start_display()
