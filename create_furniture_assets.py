"""
Create simple furniture .blend files using Blender
Run this with: blender --background --python create_furniture_assets.py
"""
import bpy
import os

# Clear the default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def create_table():
    """Create a simple table"""
    # Table top
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.75))
    table_top = bpy.context.active_object
    table_top.scale = (0.6, 0.4, 0.05)
    table_top.name = "TableTop"

    # 4 legs
    leg_positions = [
        (-0.5, -0.35, 0.375),
        (0.5, -0.35, 0.375),
        (-0.5, 0.35, 0.375),
        (0.5, 0.35, 0.375)
    ]

    for i, pos in enumerate(leg_positions):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        leg = bpy.context.active_object
        leg.scale = (0.05, 0.05, 0.375)
        leg.name = f"TableLeg{i}"

    # Select all and join
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    table = bpy.context.active_object
    table.name = "Table"

    # Save
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(), "Assets", "Furniture", "table.blend"))
    print("Created table.blend")

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_chair():
    """Create a simple chair"""
    # Seat
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.45))
    seat = bpy.context.active_object
    seat.scale = (0.4, 0.4, 0.05)
    seat.name = "ChairSeat"

    # Backrest
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.35, 0.7))
    backrest = bpy.context.active_object
    backrest.scale = (0.4, 0.05, 0.25)
    backrest.name = "ChairBack"

    # 4 legs
    leg_positions = [
        (-0.35, -0.35, 0.225),
        (0.35, -0.35, 0.225),
        (-0.35, 0.35, 0.225),
        (0.35, 0.35, 0.225)
    ]

    for i, pos in enumerate(leg_positions):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        leg = bpy.context.active_object
        leg.scale = (0.04, 0.04, 0.225)
        leg.name = f"ChairLeg{i}"

    # Join
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    chair = bpy.context.active_object
    chair.name = "Chair"

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(), "Assets", "Furniture", "chair.blend"))
    print("Created chair.blend")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_bed():
    """Create a simple bed"""
    # Mattress
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.3))
    mattress = bpy.context.active_object
    mattress.scale = (0.9, 1.9, 0.3)
    mattress.name = "Mattress"

    # Headboard
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -1.8, 0.65))
    headboard = bpy.context.active_object
    headboard.scale = (0.9, 0.1, 0.35)
    headboard.name = "Headboard"

    # Join
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    bed = bpy.context.active_object
    bed.name = "Bed"

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(), "Assets", "Furniture", "bed.blend"))
    print("Created bed.blend")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_sofa():
    """Create a simple sofa"""
    # Seat
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.3))
    seat = bpy.context.active_object
    seat.scale = (1.5, 0.8, 0.3)
    seat.name = "SofaSeat"

    # Backrest
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.7, 0.65))
    backrest = bpy.context.active_object
    backrest.scale = (1.5, 0.1, 0.35)
    backrest.name = "SofaBack"

    # Left armrest
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-1.4, 0, 0.4))
    left_arm = bpy.context.active_object
    left_arm.scale = (0.1, 0.8, 0.4)
    left_arm.name = "SofaArmLeft"

    # Right armrest
    bpy.ops.mesh.primitive_cube_add(size=1, location=(1.4, 0, 0.4))
    right_arm = bpy.context.active_object
    right_arm.scale = (0.1, 0.8, 0.4)
    right_arm.name = "SofaArmRight"

    # Join
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()
    sofa = bpy.context.active_object
    sofa.name = "Sofa"

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(), "Assets", "Furniture", "sofa.blend"))
    print("Created sofa.blend")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

# Create all furniture
create_table()
create_chair()
create_bed()
create_sofa()

print("All furniture assets created successfully!")