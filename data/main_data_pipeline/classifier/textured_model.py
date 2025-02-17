import argparse
import json
import os
import sys
import time

import bpy


def write_done(path: str, mark: str, status: bool):
    file_name = mark
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write(str(status))
                time.sleep(0.01)


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Objaverse glb mesh status calculation start. Local time is %s" %
          (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path to source mesh')
    parser.add_argument('--done_file_mark', type=str,
                        help='mark of white file')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_path = args.source_mesh_path
    mesh_folder = os.path.split(source_mesh_path)[0]
    done_file_mark = args.done_file_mark

    bpy.ops.wm.obj_import(filepath=source_mesh_path)
    meshes = []
    for ind, obj in enumerate(bpy.context.selected_objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    textured_model = True
    for mesh in meshes:
        if mesh.material_slots:
            for slot in mesh.material_slots:
                material_name = slot.material.name
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Base Color"].links) > 0:
                            textured_model = False

    if textured_model:
        print("Model at %s has texture image......." % (source_mesh_path))
    else:
        print("Model at %s do not have texture image......." % (source_mesh_path))
    write_done(mesh_folder, done_file_mark, textured_model)
