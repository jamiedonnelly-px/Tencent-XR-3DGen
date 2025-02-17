import argparse
import os
import shutil
import sys
import time

import bpy


def read_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def export_mesh(mesh, mesh_path, path_mode='STRIP', global_scale=1):
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    print("export mesh", mesh, " with %i triangles in format %s" %
          (len(mesh.data.polygons), mesh_extension.replace(".", "")))
    if mesh_extension == ".obj":
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        version_info = bpy.app.version
        if version_info[0] > 2:
            bpy.ops.wm.obj_export(filepath=mesh_path,
                                  path_mode=path_mode,
                                  export_selected_objects=True,
                                  global_scale=global_scale)
        else:
            bpy.ops.export_scene.obj(filepath=mesh_path,
                                     use_selection=True,
                                     path_mode=path_mode,
                                     global_scale=global_scale)
        bpy.ops.object.select_all(action='DESELECT')
    elif mesh_extension == ".fbx":
        bpy.ops.export_scene.fbx(filepath=mesh_path,
                                 global_scale=global_scale,
                                 use_selection=True,
                                 path_mode=path_mode)

    return mesh


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def alter_image_node_path(object, new_image_folder: str):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                material_name = slot.material.name
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Base Color"].links) > 0:
                            l = node.inputs["Base Color"].links[0]
                            if l is not None:
                                original_tex_image_node = l.from_node
                                old_image_name = original_tex_image_node.image.filepath
                                old_image_filename = os.path.split(old_image_name)[1]
                                new_image_name = os.path.join(new_image_folder,
                                                              material_name + "_" + old_image_filename)
                                shutil.copyfile(old_image_name, new_image_name)

                                texture_image = bpy.data.images.load(new_image_name)
                                diffusion_node = node_tree.nodes.new("ShaderNodeTexImage")
                                diffusion_node.image = texture_image
                                diffusion_node.image.colorspace_settings.name = "sRGB"

                                if l is not None:
                                    node_tree.links.remove(l)
                                if original_tex_image_node is not None:
                                    node_tree.nodes.remove(original_tex_image_node)

                                node_tree.links.new(diffusion_node.outputs["Color"], node.inputs["Base Color"])


def load_mesh(mesh_path: str):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_folder_name = os.path.split(mesh_folder)[1]
    mesh_parent_folder = os.path.split(mesh_folder)[0]
    mesh_parent_folder_name = os.path.split(mesh_parent_folder)[1]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    if mesh_extension == ".obj":
        version_info = bpy.app.version
        if version_info[0] > 2:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='NEGATIVE_Z', up_axis='Y')
        else:
            bpy.ops.import_scene.obj(
                filepath=mesh_path, axis_forward='-Z', axis_up='Y')
    elif mesh_extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path, use_anim=True)

    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)
    return meshes


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


def toggle_alpha_blend_mode(object, blend_method='OPAQUE'):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                slot.material.blend_method = blend_method


def remove_alpha_image(object):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Alpha"].links) > 0:
                            l = node.inputs["Alpha"].links[0]
                            if l is not None:
                                alpha_image_node = l.from_node
                                if alpha_image_node is not None:
                                    alpha_image = alpha_image_node.image
                                    node_tree.nodes.remove(alpha_image_node)
                                    bpy.data.images.remove(alpha_image)
                                # node_tree.links.remove(l)
                                node.inputs["Alpha"].default_value = 1.0


def toggle_alpha_linkage(object):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Alpha"].links) > 0:
                            l = node.inputs["Alpha"].links[0]
                            node.inputs["Alpha"].links.new(
                                l.input.outputs["Alpha"], node.inputs["Alpha"])


def fix_transparency(mesh_path: str, output_mesh_path: str, output_texture_path: str,
                     remove_alpha=False, change_blend_mode=False,
                     change_linkage=False):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    if mesh_extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif mesh_extension == ".obj":
        bpy.ops.import_scene.obj(filepath=mesh_path)

    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    textured_mesh = join_list_of_mesh(meshes)

    if remove_alpha:
        remove_alpha_image(textured_mesh)

    if change_blend_mode:
        toggle_alpha_blend_mode(textured_mesh)

    if change_linkage:
        toggle_alpha_linkage(textured_mesh)

    alter_image_node_path(textured_mesh, output_texture_path)

    export_mesh(textured_mesh, output_mesh_path)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Fix transparency process start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='input mesh file path')
    parser.add_argument('--output_mesh_path', type=str,
                        help='input mesh file path')
    parser.add_argument('--remove_alpha', action='store_true',
                        help='remove alpha linkage')
    parser.add_argument('--change_linkage', action='store_true',
                        help='add alpha linkage for mesh without alpha connection')

    args = parser.parse_args(raw_argv)
    output_mesh_path = args.output_mesh_path
    mesh_path = args.mesh_path

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    output_mesh_folder = os.path.split(output_mesh_path)[0]
    output_texture_folder = os.path.join(output_mesh_folder, "texture")
    if not os.path.exists(output_texture_folder):
        os.mkdir(output_texture_folder)

    fix_transparency(mesh_path, output_mesh_path,
                     output_texture_folder,
                     remove_alpha=args.remove_alpha,
                     change_linkage=args.change_linkage)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Fix transparency process done. Local time is %s" % (local_time_str))
