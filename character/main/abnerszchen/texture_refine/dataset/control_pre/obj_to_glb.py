import os
import sys

import bpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument(
    "--output_glb_path",
    type=str,
    required=True,
    help="Path to the output glb file",
)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=args.input_path)

imported_objs = bpy.context.selected_objects

for obj in imported_objs:
    obj.select_set(True)

os.makedirs(os.path.dirname(args.output_glb_path), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=args.output_glb_path, export_format='GLB', use_selection=True)