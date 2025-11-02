"""
Quick test to see if Blender can import furniture
"""
import bpy
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Try to import a table
furniture_path = os.path.join(os.getcwd(), "Assets", "Furniture", "table.blend")
print(f"Trying to import from: {furniture_path}")
print(f"File exists: {os.path.exists(furniture_path)}")

if os.path.exists(furniture_path):
    try:
        with bpy.data.libraries.load(furniture_path, link=False) as (data_from, data_to):
            print(f"Objects in file: {data_from.objects}")
            data_to.objects = data_from.objects
        
        # Link to scene
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
                print(f"Imported object: {obj.name}")
        
        print("SUCCESS: Furniture imported!")
    except Exception as e:
        print(f"ERROR: {e}")
else:
    print("ERROR: File not found!")

# Save test file
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.getcwd(), "test_import.blend"))
print("Saved to test_import.blend")
