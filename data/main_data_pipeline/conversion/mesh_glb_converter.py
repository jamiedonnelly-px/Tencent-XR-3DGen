import argparse
import os
import sys
import time

import bpy


def join_list_of_mesh(mesh_list):
    assert len(mesh_list) > 0
    if len(mesh_list) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for ind, obj in enumerate(mesh_list):
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
        joint_mesh = bpy.context.object
    else:
        joint_mesh = mesh_list[0]
    return joint_mesh


def load_mesh(mesh_path: str):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    mesh_extension_lower = mesh_extension.lower()
    if mesh_extension_lower == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path, use_anim=True)
    elif mesh_extension_lower == ".obj":
        version_info = bpy.app.version
        if version_info[0] > 2:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        else:
            bpy.ops.import_scene.obj(filepath=mesh_path)
    elif mesh_extension_lower == '.blend':
        bpy.ops.wm.open_mainfile(filepath=mesh_path)


def export_glb(mesh, mesh_path: str):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.export_scene.gltf(filepath=mesh_path, use_selection=True)
    time.sleep(0.1)


def toggle_alpha_blend_mode(object, blend_method='OPAQUE'):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                slot.material.blend_method = blend_method


if __name__ == '__main__':

    t_start = time.time()
    local_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("obj/fbx/blend to glb start. Local time is %s" % (start_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Render data script.')
    parser.add_argument('--mesh_path', type=str,
                        help='path of mesh obj/fbx file')
    parser.add_argument('--output_fullpath', type=str,
                        help='output path of generated glb')
    parser.add_argument('--remove_color_attributes', action='store_true',
                        help='remove all color attributes in glb file')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_path = args.mesh_path
    output_fullpath = args.output_fullpath
    remove_color_attributes = args.remove_color_attributes

    load_mesh(mesh_path)

    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    joint_mesh = join_list_of_mesh(meshes)
    toggle_alpha_blend_mode(joint_mesh, blend_method='OPAQUE')
    time.sleep(0.1)

    if remove_color_attributes:
        if joint_mesh.data.color_attributes:
            attrs = joint_mesh.data.color_attributes
            for r in range(len(joint_mesh.data.color_attributes) - 1, -1, -1):
                attrs.remove(attrs[r])

    export_glb(joint_mesh, output_fullpath)

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("obj/fbx to glb finish. Start local time is %s, end local time is %s............" % (
        start_time_str, end_time_str))
