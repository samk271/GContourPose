import bpy
import os
import re

# Set the output directory
output_directory = r"C:\Users\sklemic\Documents\IREP_Summer_2024\ContourPose\GContourPose\data\set2\set2\scene3\renders"

# Get all objects in the scene
objects = bpy.context.scene.objects

# Filter out cameras
objects_to_render = [obj for obj in objects if obj.type == 'MESH']

# Loop through each object, unhide and render
for i, obj_to_render in enumerate(objects_to_render): 
    directory_name = re.split('[A-Z]', obj_to_render.data.name)[0][:-1] + "\\"
    # Set the output file path
    output_folder = os.path.join(output_directory, directory_name)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Hide all objects 
    for obj in objects_to_render:
        obj.hide_render = True

    # Unhide the current object
    obj_to_render.hide_render = False

    # Set the output file path
    bpy.context.scene.render.filepath = output_folder
    # Render the scene
    print("Rendering " + directory_name)
    bpy.ops.render.render(animation=True)

print("Rendering completed.")