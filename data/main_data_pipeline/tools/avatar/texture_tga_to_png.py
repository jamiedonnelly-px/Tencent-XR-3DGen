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


def fix_bump_color_space(object):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name
            nodes = node_tree.nodes
            links = node_tree.links

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(node.inputs)
                    if len(node.inputs["Normal"].links) > 0:
                        l = node.inputs["Normal"].links[0]
                        if l.from_socket.name == 'Normal':
                            normal_vector_node = l.from_node
                            if len(normal_vector_node.inputs["Color"].links) > 0:
                                l_bump = normal_vector_node.inputs["Color"].links[0]
                                if l_bump.from_socket.name == 'Color':
                                    bump_iamge_node = l_bump.from_node
                                    bump_iamge_node.image.colorspace_settings.name = "Non-Color"


def fix_material_space(object, input_type="Metallic"):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(node.inputs)
                    if len(node.inputs[input_type].links) > 0:
                        l = node.inputs[input_type].links[0]
                        if l.from_socket.name == 'Color':
                            material_image_node = l.from_node
                            material_image_node.image.colorspace_settings.name = "Non-Color"


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, z_up=False):
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


def change_texture_image_to_png(object, color_type: str = "sRGB", alpha_mode='NONE'):
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if len(node.inputs["Base Color"].links) > 0:
                        l = node.inputs["Base Color"].links[0]
                        if l.from_socket.name == 'Color':
                            tex_image_node = l.from_node
                            if tex_image_node.type == 'TEX_IMAGE':
                                old_image = tex_image_node.image.copy()
                                old_image.alpha_mode = alpha_mode
                                old_image_filepath = old_image.filepath
                                new_image_folder = os.path.split(
                                    old_image_filepath)[0]
                                old_image_filename = os.path.split(
                                    old_image_filepath)[1]
                                old_image_basename = os.path.splitext(
                                    old_image_filename)[0]
                                new_image_filename = os.path.join(
                                    new_image_folder, old_image_basename + ".png")
                                print(old_image_filepath, new_image_filename)
                                # old_image.filepath_raw = new_image_filename
                                # old_image.file_format = 'PNG'
                                old_image.save_render(new_image_filename)

                                new_texture_image = bpy.data.images.load(
                                    new_image_filename)
                                new_diffusion_node = nodes.new(
                                    "ShaderNodeTexImage")
                                new_diffusion_node.image = new_texture_image
                                new_diffusion_node.image.colorspace_settings.name = color_type
                                node_tree.links.remove(l)
                                nodes.remove(tex_image_node)
                                links.new(
                                    new_diffusion_node.outputs["Color"], node.inputs["Base Color"])


def check_mesh_with_certain_tex_image(object, tex_image_name: str = 'material'):
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if len(node.inputs["Base Color"].links) > 0:
                        l = node.inputs["Base Color"].links[0]
                        if l.from_socket.name == 'Color':
                            tex_image_node = l.from_node
                            if tex_image_node.type == 'TEX_IMAGE':
                                if tex_image_name in tex_image_node.image.filepath:
                                    return True
    return False


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
                                    # bpy.data.images.remove(alpha_image)
                                # node_tree.links.remove(l)
                                node.inputs["Alpha"].default_value = 1.0


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to imported mesh')
    parser.add_argument('--output_fullpath', type=str,
                        default="", help='mesh result output folder')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_fullpath = args.output_fullpath

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.context.scene.view_settings.view_transform = 'Standard'

    meshes = load_mesh(mesh_path)
    joint_mesh = join_list_of_mesh(meshes)

    bpy.ops.object.mode_set(mode='EDIT')
    # Seperate by material
    bpy.ops.mesh.separate(type='MATERIAL')
    # Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    correct_mesh_list = []
    to_delete_mesh_list = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            eyeline_result = check_mesh_with_certain_tex_image(
                obj, tex_image_name='eyeline')
            eyeshade_result = check_mesh_with_certain_tex_image(
                obj, tex_image_name='eyeshade')
            if eyeline_result or eyeshade_result:
                to_delete_mesh_list.append(obj)
            else:
                correct_mesh_list.append(obj)

    correct_mesh = join_list_of_mesh(correct_mesh_list)
    for skin_mesh in to_delete_mesh_list:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = skin_mesh
        skin_mesh.select_set(True)
        bpy.ops.object.delete(use_global=False)

    cleaned_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            cleaned_meshes.append(obj)
    cleaned_joint_mesh = join_list_of_mesh(cleaned_meshes)

    remove_alpha_image(cleaned_joint_mesh)
    change_texture_image_to_png(cleaned_joint_mesh, alpha_mode='NONE')

    time.sleep(0.1)

    export_mesh_obj(cleaned_joint_mesh, output_fullpath,
                    path_mode='COPY', z_up=False)
