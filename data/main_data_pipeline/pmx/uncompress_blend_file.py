import os
import numpy as np
import argparse
import time
import math
import json
import multiprocessing
import bpy
import bmesh
import sys

weapon = ["weapon", "Weapon"]

if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(
        description='Uncompress a compressed .blend file.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to .blend file of pmx mesh to be converted')
    parser.add_argument('--uncompressed_mesh_path', type=str,
                        help='path to uncompressed .blend file')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    uncompressed_mesh_path = args.uncompressed_mesh_path

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_folder = os.path.split(mesh_path)[0]
    mesh_name = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_name)[0]
    mesh_type = os.path.splitext(mesh_name)[1]

    bpy.ops.wm.open_mainfile(filepath=mesh_path)
    time.sleep(0.1)

    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    size_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            if any(wp in obj.name for wp in weapon):
                obj.select_set(state=True)
                bpy.ops.object.delete()
            else:
                meshes.append(obj)

    bpy.ops.wm.save_as_mainfile(
        filepath=uncompressed_mesh_path, compress=False)

    print("%s blend file change to incompressed version at %s" %
          (mesh_path, uncompressed_mesh_path))
