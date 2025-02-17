import os
import numpy as np
import argparse
import time
import math
import json
import multiprocessing
import bpy
import sys

rigid_noise_flag = ["rigid", "Rigid"]

if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(
        description='Uncompress a compressed .blend file.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to .obj file from pmx mesh')
    parser.add_argument('--denoise_mesh_path', type=str,
                        help='path to denoised .obj file')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    denoise_mesh_path = args.denoise_mesh_path

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_folder = os.path.split(mesh_path)[0]
    mesh_name = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_name)[0]
    mesh_type = os.path.splitext(mesh_name)[1]

    try:
        bpy.ops.import_scene.obj(filepath=mesh_path)
        bpy.ops.object.select_all(action='DESELECT')
        meshes = []
        size_meshes = []
        for ind, obj in enumerate(bpy.context.scene.objects):
            if obj.type == 'MESH':
                print(obj.name)
                if any(noise in obj.name for noise in rigid_noise_flag):
                    obj.select_set(state=True)
                    bpy.ops.object.delete()
                else:
                    meshes.append(obj)
    except:
        pass

    time.sleep(0.1)

    bpy.ops.export_scene.obj(filepath=denoise_mesh_path, path_mode='RELATIVE')
