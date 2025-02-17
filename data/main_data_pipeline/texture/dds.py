import argparse
import os
import sys
import time

import bpy


def load_mesh(mesh_path: str, z_up=False):
    bpy.ops.object.select_all(action='DESELECT')
    version_info = bpy.app.version
    if version_info[0] > 2:
        if z_up:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='Y', up_axis='Z')
        else:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        bpy.ops.import_scene.obj(
            filepath=mesh_path, axis_forward='-Z', axis_up='Y')
    meshes = []
    for ind, obj in enumerate(bpy.context.selected_objects):
        if obj.type == 'MESH':
            meshes.append(obj)
    return meshes


def join_list_of_mesh(mesh_list: list):
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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              global_scale=global_scale,
                              export_selected_objects=True)
    else:
        bpy.ops.export_scene.obj(filepath=mesh_path,
                                 use_selection=True,
                                 path_mode=path_mode,
                                 global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


def convert_dds(object):
    bpy.context.scene.view_settings.view_transform = 'Standard'
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name
            nodes = node_tree.nodes
            links = node_tree.links

            for node in node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    tex_image_path = node.image.filepath
                    tex_image_basename = os.path.splitext(tex_image_path)[0]
                    tex_image_extension = os.path.splitext(tex_image_path)[1]
                    if tex_image_extension == ".dds":
                        new_tex_image_path = os.path.join(tex_image_basename, ".png")
                        node.image.save_render(new_tex_image_path)
                        new_texture_image_data_block = bpy.data.images.load(new_tex_image_path)
                        node.image = new_texture_image_data_block


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("DDS conversion start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to source mesh')
    parser.add_argument('--output_mesh_path', type=str, default="",
                        help='path to output mesh with dds converted.')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_path = args.mesh_path
    output_mesh_path = args.output_mesh_path
    copy_texture = args.copy_texture

    mesh_data_list = load_mesh(mesh_path)
    joint_mesh = join_list_of_mesh(mesh_data_list)

    convert_dds(joint_mesh)
    if copy_texture:
        export_mesh_obj(joint_mesh, output_mesh_path, path_mode='COPY')
    else:
        export_mesh_obj(joint_mesh, output_mesh_path, path_mode='RELATIVE')
