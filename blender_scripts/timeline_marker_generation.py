import bpy
import sys

scene = bpy.data.scenes['Scene']
frame_count = 1
print(bpy.data.objects)

for object in bpy.data.objects:
    print(object.name, "::", frame_count, file=sys.stderr)
    if object.type == 'CAMERA':
        object.select_set(True)
        marker = scene.timeline_markers.new(object.name, frame=frame_count)
        marker.camera = object
        frame_count += 1