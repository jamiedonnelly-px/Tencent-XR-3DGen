import os
import argparse
import time
import math
import json
import bpy
import bmesh
import sys

weapon = ["weapon", "Weapon"]


def pmx_import(pmx_path: str, blend_filepath: str):
    bpy.ops.mmd_tools.import_model(filepath=pmx_path)
    time.sleep(0.1)
    # bpy.ops.cats_armature.fix()
    bpy.ops.wm.save_as_mainfile(filepath=blend_filepath, compress=False)


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(
        description='Uncompress a compressed .blend file.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to .blend file of pmx mesh to be converted')
    parser.add_argument('--obj_mesh_file', type=str,
                        help='path of output obj meshfile')
    args = parser.parse_args(raw_argv)

    bpy.ops.preferences.addon_enable(module="cats-blender-plugin-master")
    bpy.ops.preferences.addon_enable(module="material-combiner-addon-master")

    mesh_path = args.mesh_path
    obj_mesh_file = args.obj_mesh_file

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_folder = os.path.split(mesh_path)[0]
    mesh_name = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_name)[0]
    mesh_type = os.path.splitext(mesh_name)[1]

    obj_mesh_folder = os.path.split(obj_mesh_file)[0]
    atlas_folder = os.path.join(obj_mesh_folder, "atlas")
    if not os.path.exists(atlas_folder):
        os.mkdir(atlas_folder)

    try:
        bpy.ops.wm.open_mainfile(filepath=mesh_path)
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
    except:
        pass

    bpy.ops.smc.refresh_ob_data()
    bpy.ops.smc.combiner(directory=atlas_folder)
    bpy.data.materials["material_atlas_00001_1"].node_tree.nodes["Principled BSDF"].inputs[
        'Specular'].default_value = 0.0

    time.sleep(0.1)

    bpy.ops.export_scene.obj(filepath=obj_mesh_file, path_mode='RELATIVE')
    print("Export blend project (from imported pmx file) from %s to %s" %
          (mesh_path, obj_mesh_file))
