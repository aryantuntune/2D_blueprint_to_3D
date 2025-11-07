import bpy
import numpy as np
import json
import sys
import math
import os.path

# TODO: restructure this file with a class and help-function to save a lot of lines of code!
# TODO: fix index should be same as floorplan folder


def read_from_file(file_path):

    # Now read the file back into a Python list object
    with open(file_path + ".txt", "r") as f:
        data = json.loads(f.read())
    return data


def init_object(name):
    # Create new blender object and return references to mesh and object
    mymesh = bpy.data.meshes.new(name)
    myobject = bpy.data.objects.new(name, mymesh)
    bpy.context.collection.objects.link(myobject)
    return myobject, mymesh


def average(lst):
    return sum(lst) / len(lst)


def get_mesh_center(verts):
    # Calculate center location of a mesh from verts
    x = []
    y = []
    z = []

    for vert in verts:
        x.append(vert[0])
        y.append(vert[1])
        z.append(vert[2])

    return [average(x), average(y), average(z)]


def subtract_center_verts(verts1, verts2):
    # Remove verts1 from all verts in verts2, return result, verts1 & verts2 must have same shape!
    for i in range(0, len(verts2)):
        verts2[i][0] -= verts1[0]
        verts2[i][1] -= verts1[1]
        verts2[i][2] -= verts1[2]
    return verts2


def create_custom_mesh(objname, verts, faces, mat=None, cen=None):
    """
    @Param objname, name of new mesh
    @Param pos, object position [x, y, z]
    @Param vertex, corners
    @Param faces, buildorder
    """
    # Create mesh and object
    myobject, mymesh = init_object(objname)

    # Rearrange verts to put pivot point in center of mesh
    # Find center of verts
    center = get_mesh_center(verts)
    # Subtract center from verts before creation
    proper_verts = subtract_center_verts(center, verts)

    # Generate mesh data
    mymesh.from_pydata(proper_verts, [], faces)
    # Calculate the edges
    mymesh.update(calc_edges=True)

    parent_center = [0, 0, 0]
    if cen is not None:
        parent_center = [int(cen[0] / 2), int(cen[1] / 2), int(cen[2])]

    # Move object to input verts location
    myobject.location.x = center[0] - parent_center[0]
    myobject.location.y = center[1] - parent_center[1]
    myobject.location.z = center[2] - parent_center[2]

    # add material
    if mat is None:  # add random color
        myobject.data.materials.append(
            create_mat(np.random.randint(0, 40, size=4))
        )  # add the material to the object
    else:
        myobject.data.materials.append(mat)  # add the material to the object
    return myobject


def create_mat(rgb_color):
    mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
    mat.diffuse_color = rgb_color  # change to random color
    return mat


def import_furniture_model(model_name, location, size, parent, cen, program_path):
    """Import a furniture model from Assets folder and place it at location"""
    import os

    # Build furniture path using program_path
    furniture_path = os.path.join(program_path, "Assets", "Furniture", f"{model_name}.blend")

    # Check if furniture file exists
    if not os.path.exists(furniture_path):
        print(f"Warning: {model_name}.blend not found at {furniture_path}")
        return

    # Calculate position
    parent_center = [0, 0, 0]
    if cen is not None:
        parent_center = [int(cen[0] / 2), int(cen[1] / 2), int(cen[2])]

    x = location[0] - parent_center[0]
    y = location[1] - parent_center[1]
    z = 0  # Place furniture on the floor

    # Import the furniture model
    try:
        print(f"=== IMPORTING {model_name} ===")
        print(f"Path: {furniture_path}")
        print(f"File exists: {os.path.exists(furniture_path)}")

        with bpy.data.libraries.load(furniture_path, link=False) as (data_from, data_to):
            print(f"Objects in blend file: {data_from.objects}")
            data_to.objects = data_from.objects

        # Link imported objects to scene and position them
        imported_count = 0
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)

                # Intelligent scaling based on detected furniture size FIRST
                # Size is in pixels from the blueprint detection
                width = size[0]
                height = size[1]
                avg_size = (width + height) / 2

                # Scale furniture proportionally to detected size
                # Base scale calculation: detected size / reference size
                # Reference: typical furniture is ~50 pixels in blueprint
                base_scale = avg_size / 50.0

                # Clamp scale to reasonable range (0.4 to 2.0)
                scale_factor = max(0.4, min(2.0, base_scale))

                obj.scale = (scale_factor, scale_factor, scale_factor)

                # Rotate furniture to align with floorplan orientation
                # Rotate +90 on Z-axis so furniture faces +Y direction
                obj.rotation_euler[0] = math.radians(0)    # No X-axis flip
                obj.rotation_euler[1] = math.radians(0)    # No Y-axis rotation
                obj.rotation_euler[2] = math.radians(90)   # Rotate +90 degrees to face +Y direction

                # Position furniture ON the floor surface
                # The floor surface is at the bottom of walls (Z = -1.0)
                # Get the mesh's local Z bounds to place it correctly
                if obj.data and obj.data.vertices:
                    z_min = min([v.co.z for v in obj.data.vertices])
                    # Offset to compensate for mesh origin - place bottom at floor level
                    z_offset = -z_min  # This lifts the furniture so its bottom is at object origin
                    obj.location = (x, y, -1.0 + z_offset)  # Place furniture at floor level
                else:
                    obj.location = (x, y, -1.0)

                # Set parent
                obj.parent = parent

                imported_count += 1
                print(f"✓ Imported {obj.name} at ({x:.2f}, {y:.2f}, {z:.2f}) scale={scale_factor:.3f}")

        if imported_count == 0:
            print(f"WARNING: No objects imported from {model_name}.blend")
    except Exception as e:
        print(f"ERROR importing {model_name}: {e}")
        import traceback
        traceback.print_exc()


# Global variable to store program_path
_program_path = None

def set_program_path(path):
    global _program_path
    _program_path = path


# Wrapper functions that call the import function
def create_table_3d(location, size, parent, cen):
    """Import table model"""
    import_furniture_model("table", location, size, parent, cen, _program_path)


def create_bed_3d(location, size, parent, cen):
    """Import bed model"""
    import_furniture_model("bed", location, size, parent, cen, _program_path)


def create_chair_3d(location, size, parent, cen):
    """Import chair model"""
    import_furniture_model("chair", location, size, parent, cen, _program_path)


def create_sofa_3d(location, size, parent, cen):
    """Import sofa model"""
    import_furniture_model("sofa", location, size, parent, cen, _program_path)


def create_kitchen_item_3d(location, size, parent, cen):
    """Create simple kitchen counter"""
    width = size[0]
    depth = size[1]
    counter_height = 90

    parent_center = [0, 0, 0]
    if cen is not None:
        parent_center = [int(cen[0] / 2), int(cen[1] / 2), int(cen[2])]

    x = location[0] - parent_center[0]
    y = location[1] - parent_center[1]
    z = 0

    bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z + counter_height/2))
    counter = bpy.context.active_object
    counter.scale = (width/2, depth/2, counter_height/2)
    counter.data.materials.append(create_mat((0.7, 0.7, 0.7, 1)))  # Grey counter
    counter.parent = parent


"""
Main functionality here!
"""


def main(argv):

    # Remove starting object cube
    objs = bpy.data.objects
    objs.remove(objs["Cube"], do_unlink=True)

    if len(argv) > 7:  # Note YOU need 8 arguments!
        program_path = argv[5]
        target = argv[6]
    else:
        exit(0)

    """
    Instantiate
    Each argument after 7 will be a floorplan path
    """
    for i in range(7, len(argv)):
        base_path = argv[i]
        create_floorplan(base_path, program_path, i)

    """
    Save to file
    TODO add several save modes here!
    """
    bpy.ops.wm.save_as_mainfile(filepath=program_path + target)  # "/floorplan.blend"

    """
    Send correct exit code
    """
    exit(0)


def create_floorplan(base_path, program_path, name=None):

    if name is None:
        name = 0

    # Set program path for furniture import
    set_program_path(program_path)

    parent, _ = init_object("Floorplan" + str(name))

    """
    Get transform data
    """

    path_to_transform_file = program_path + "/" + base_path + "transform"

    # read from file
    transform = read_from_file(path_to_transform_file)

    rot = transform["rotation"]
    pos = transform["position"]
    scale = transform["scale"]

    # Calculate and move floorplan shape to center
    cen = transform["shape"]

    # Where data is stored, if shared between floorplans
    path_to_data = transform["origin_path"]

    # Set Cursor start
    bpy.context.scene.cursor.location = (0, 0, 0)

    path_to_wall_vertical_faces_file = (
        program_path + "/" + path_to_data + "wall_vertical_faces"
    )
    path_to_wall_vertical_verts_file = (
        program_path + "/" + path_to_data + "wall_vertical_verts"
    )

    path_to_wall_horizontal_faces_file = (
        program_path + "/" + path_to_data + "wall_horizontal_faces"
    )
    path_to_wall_horizontal_verts_file = (
        program_path + "/" + path_to_data + "wall_horizontal_verts"
    )

    path_to_floor_faces_file = program_path + "/" + path_to_data + "floor_faces"
    path_to_floor_verts_file = program_path + "/" + path_to_data + "floor_verts"

    path_to_rooms_faces_file = program_path + "/" + path_to_data + "room_faces"
    path_to_rooms_verts_file = program_path + "/" + path_to_data + "room_verts"

    path_to_doors_vertical_faces_file = (
        program_path + "\\" + path_to_data + "door_vertical_faces"
    )
    path_to_doors_vertical_verts_file = (
        program_path + "\\" + path_to_data + "door_vertical_verts"
    )

    path_to_doors_horizontal_faces_file = (
        program_path + "\\" + path_to_data + "door_horizontal_faces"
    )
    path_to_doors_horizontal_verts_file = (
        program_path + "\\" + path_to_data + "door_horizontal_verts"
    )

    path_to_windows_vertical_faces_file = (
        program_path + "\\" + path_to_data + "window_vertical_faces"
    )
    path_to_windows_vertical_verts_file = (
        program_path + "\\" + path_to_data + "window_vertical_verts"
    )

    path_to_windows_horizontal_faces_file = (
        program_path + "\\" + path_to_data + "window_horizontal_faces"
    )
    path_to_windows_horizontal_verts_file = (
        program_path + "\\" + path_to_data + "window_horizontal_verts"
    )

    """
    Create Walls
    """

    if (
        os.path.isfile(path_to_wall_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_wall_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_wall_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_wall_horizontal_faces_file + ".txt")
    ):
        # get image wall data
        verts = read_from_file(path_to_wall_vertical_verts_file)
        faces = read_from_file(path_to_wall_vertical_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        # Create parent
        wall_parent, _ = init_object("Walls")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.5, 0.5, 0.5, 1)),
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        # get image top wall data
        verts = read_from_file(path_to_wall_horizontal_verts_file)
        faces = read_from_file(path_to_wall_horizontal_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertWalls" + str(i)
            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.5, 0.5, 0.5, 1)),
            )
            obj.parent = wall_parent

        wall_parent.parent = parent

    """
    Create Windows
    """
    if (
        os.path.isfile(path_to_windows_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_windows_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_windows_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_windows_horizontal_faces_file + ".txt")
    ):
        # get image wall data
        verts = read_from_file(path_to_windows_vertical_verts_file)
        faces = read_from_file(path_to_windows_vertical_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        # Create parent
        wall_parent, _ = init_object("Windows")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.7, 0.9, 1.0, 0.6)),  # Light blue transparent for windows
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        # get windows
        verts = read_from_file(path_to_windows_horizontal_verts_file)
        faces = read_from_file(path_to_windows_horizontal_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertWindow" + str(i)
            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.7, 0.9, 1.0, 0.6)),  # Light blue transparent for windows
            )
            obj.parent = wall_parent

        wall_parent.parent = parent

    """
    Create Doors
    """
    if (
        os.path.isfile(path_to_doors_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_doors_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_doors_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_doors_horizontal_faces_file + ".txt")
    ):

        # get image wall data
        verts = read_from_file(path_to_doors_vertical_verts_file)
        faces = read_from_file(path_to_doors_vertical_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        # Create parent
        wall_parent, _ = init_object("Doors")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.6, 0.4, 0.2, 1)),  # Brown for doors
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        # get windows
        verts = read_from_file(path_to_doors_horizontal_verts_file)
        faces = read_from_file(path_to_doors_horizontal_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertDoor" + str(i)
            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.6, 0.4, 0.2, 1)),  # Brown for doors
            )
            obj.parent = wall_parent

        wall_parent.parent = parent

    """
    Create Floor
    """
    if os.path.isfile(path_to_floor_verts_file + ".txt") and os.path.isfile(
        path_to_floor_faces_file + ".txt"
    ):

        # get image wall data
        verts = read_from_file(path_to_floor_verts_file)
        faces = read_from_file(path_to_floor_faces_file)

        # Create mesh from data
        cornername = "Floor"
        obj = create_custom_mesh(
            cornername, verts, [faces], mat=create_mat((40, 1, 1, 1)), cen=cen
        )
        obj.parent = parent

        """
        Create rooms
        """
        # get image wall data
        verts = read_from_file(path_to_rooms_verts_file)
        faces = read_from_file(path_to_rooms_faces_file)

        # Create parent
        room_parent, _ = init_object("Rooms")

        for i in range(0, len(verts)):
            roomname = "Room" + str(i)
            obj = create_custom_mesh(roomname, verts[i], faces[i], cen=cen)
            obj.parent = room_parent

        room_parent.parent = parent

    """
    Create Furniture (Tables, Beds, Chairs, Sofas, Kitchen Items) as Proper 3D Models
    """
    # Tables - Create individual table objects with legs
    path_to_table_verts_file = program_path + "/" + path_to_data + "table_verts"
    path_to_table_faces_file = program_path + "/" + path_to_data + "table_faces"

    if os.path.isfile(path_to_table_verts_file + ".txt") and os.path.isfile(path_to_table_faces_file + ".txt"):
        verts = read_from_file(path_to_table_verts_file)
        faces = read_from_file(path_to_table_faces_file)

        if len(verts) > 0 and len(faces) > 0:
            furniture_parent, _ = init_object("Tables")
            furniture_parent.parent = parent

            # Each table is represented by 8 vertices (a box)
            # Group them by sets of 8 to create individual tables
            for i in range(0, len(verts), 8):
                if i + 7 < len(verts):
                    # Extract the 8 vertices for this table
                    table_verts = verts[i:i+8]

                    # Calculate bounding box
                    x_coords = [v[0] for v in table_verts]
                    y_coords = [v[1] for v in table_verts]

                    x_center = sum(x_coords) / len(x_coords)
                    y_center = sum(y_coords) / len(y_coords)
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)

                    # Create proper 3D table with legs
                    create_table_3d([x_center, y_center], [width, height], furniture_parent, cen)

    # Beds - Create individual bed objects with mattress and headboard
    path_to_bed_verts_file = program_path + "/" + path_to_data + "bed_verts"
    path_to_bed_faces_file = program_path + "/" + path_to_data + "bed_faces"

    if os.path.isfile(path_to_bed_verts_file + ".txt") and os.path.isfile(path_to_bed_faces_file + ".txt"):
        verts = read_from_file(path_to_bed_verts_file)
        faces = read_from_file(path_to_bed_faces_file)

        if len(verts) > 0 and len(faces) > 0:
            furniture_parent, _ = init_object("Beds")
            furniture_parent.parent = parent

            # Each bed is represented by 8 vertices (a box)
            for i in range(0, len(verts), 8):
                if i + 7 < len(verts):
                    bed_verts = verts[i:i+8]

                    x_coords = [v[0] for v in bed_verts]
                    y_coords = [v[1] for v in bed_verts]

                    x_center = sum(x_coords) / len(x_coords)
                    y_center = sum(y_coords) / len(y_coords)
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)

                    # Create proper 3D bed with mattress and headboard
                    create_bed_3d([x_center, y_center], [width, height], furniture_parent, cen)

    # Chairs
    path_to_chair_verts_file = program_path + "/" + path_to_data + "chair_verts"
    path_to_chair_faces_file = program_path + "/" + path_to_data + "chair_faces"

    if os.path.isfile(path_to_chair_verts_file + ".txt") and os.path.isfile(path_to_chair_faces_file + ".txt"):
        verts = read_from_file(path_to_chair_verts_file)
        faces = read_from_file(path_to_chair_faces_file)

        if len(verts) > 0 and len(faces) > 0:
            furniture_parent, _ = init_object("Chairs")

            # Each chair is represented by 8 vertices (a box)
            for i in range(0, len(verts), 8):
                if i + 7 < len(verts):
                    chair_verts = verts[i:i+8]
                    x_coords = [v[0] for v in chair_verts]
                    y_coords = [v[1] for v in chair_verts]
                    x_center = sum(x_coords) / len(x_coords)
                    y_center = sum(y_coords) / len(y_coords)
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    create_chair_3d([x_center, y_center], [width, height], furniture_parent, cen)

            furniture_parent.parent = parent

    # Sofas
    path_to_sofa_verts_file = program_path + "/" + path_to_data + "sofa_verts"
    path_to_sofa_faces_file = program_path + "/" + path_to_data + "sofa_faces"

    if os.path.isfile(path_to_sofa_verts_file + ".txt") and os.path.isfile(path_to_sofa_faces_file + ".txt"):
        verts = read_from_file(path_to_sofa_verts_file)
        faces = read_from_file(path_to_sofa_faces_file)

        if len(verts) > 0 and len(faces) > 0:
            furniture_parent, _ = init_object("Sofas")

            # Each sofa is represented by 8 vertices (a box)
            for i in range(0, len(verts), 8):
                if i + 7 < len(verts):
                    sofa_verts = verts[i:i+8]
                    x_coords = [v[0] for v in sofa_verts]
                    y_coords = [v[1] for v in sofa_verts]
                    x_center = sum(x_coords) / len(x_coords)
                    y_center = sum(y_coords) / len(y_coords)
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    create_sofa_3d([x_center, y_center], [width, height], furniture_parent, cen)

            furniture_parent.parent = parent

    # Kitchen Items
    path_to_kitchen_verts_file = program_path + "/" + path_to_data + "kitchen_verts"
    path_to_kitchen_faces_file = program_path + "/" + path_to_data + "kitchen_faces"

    if os.path.isfile(path_to_kitchen_verts_file + ".txt") and os.path.isfile(path_to_kitchen_faces_file + ".txt"):
        verts = read_from_file(path_to_kitchen_verts_file)
        faces = read_from_file(path_to_kitchen_faces_file)

        if len(verts) > 0 and len(faces) > 0:
            furniture_parent, _ = init_object("KitchenItems")

            # Each kitchen item is represented by 8 vertices (a box)
            for i in range(0, len(verts), 8):
                if i + 7 < len(verts):
                    kitchen_verts = verts[i:i+8]
                    x_coords = [v[0] for v in kitchen_verts]
                    y_coords = [v[1] for v in kitchen_verts]
                    x_center = sum(x_coords) / len(x_coords)
                    y_center = sum(y_coords) / len(y_coords)
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    create_kitchen_item_3d([x_center, y_center], [width, height], furniture_parent, cen)

            furniture_parent.parent = parent

    # Perform Floorplan final position, rotation and scale
    if rot is not None:
        # compensate for mirrored image
        parent.rotation_euler = [
            math.radians(rot[0]) + math.pi,
            math.radians(rot[1]),
            math.radians(rot[2]),
        ]

    if pos is not None:
        parent.location.x += pos[0]
        parent.location.y += pos[1]
        parent.location.z += pos[2]

    if scale is not None:
        parent.scale.x = scale[0]
        parent.scale.y = scale[1]
        parent.scale.z = scale[2]


if __name__ == "__main__":
    main(sys.argv)
