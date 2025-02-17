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


def remove_node_by_name(object, node_name: str):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.name == node_name:
                        node_tree.nodes.remove(node)


def set_node_default_value(material, toset_value, material_input_type: str = "Base Color"):
    node_tree = material.node_tree
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs[material_input_type].default_value = toset_value


def move_texture_node(material, tex_image_node, material_input_type: str = "Base Color"):
    tex_image_path = tex_image_node.image.filepath

    node_tree = material.node_tree
    material_name = material.name

    new_emission_texture_image = bpy.data.images.load(tex_image_path)
    new_emission_image_node = node_tree.nodes.new("ShaderNodeTexImage")
    new_emission_image_node.image = new_emission_texture_image

    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            if len(node.inputs[material_input_type].links) > 0:
                l = node.inputs[material_input_type].links[0]
                original_tex_image_node = l.from_node
            if l is not None:
                node_tree.links.remove(l)
            if original_tex_image_node is not None:
                node_tree.nodes.remove(original_tex_image_node)

            node_tree.links.new(
                new_emission_image_node.outputs["Color"], node.inputs[material_input_type])
            if material_input_type == 'Base Color':
                node_tree.links.new(
                    new_emission_image_node.outputs["Alpha"], node.inputs["Alpha"])


def remove_image_linkage(object, shader_name: str, material_input_type: str = "Roughness"):
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in node_tree.nodes:
                if node.name == shader_name:
                    if len(node.inputs[material_input_type].links) > 0:
                        l = node.inputs[material_input_type].links[0]
                        original_tex_image_node = l.from_node
                        if l is not None:
                            links.remove(l)
                        if original_tex_image_node is not None:
                            nodes.remove(original_tex_image_node)
                    if isinstance(node.inputs[material_input_type].default_value, float):
                        node.inputs[material_input_type].default_value = 0
                    else:
                        node.inputs[material_input_type].default_value = (
                            0, 0, 0, 1)


def pmx_import(pmx_path: str, blend_filepath: str):
    bpy.ops.mmd_tools.import_model(filepath=pmx_path)
    time.sleep(0.1)
    bpy.context.object.mmd_root.use_toon_texture = False
    bpy.context.object.mmd_root.use_sphere_texture = False
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.mmd_tools.convert_materials()
    for object_index, object_entity in enumerate(bpy.context.scene.objects):
        if object_entity.type == 'MESH':
            if object_entity.material_slots:
                for slot in object_entity.material_slots:
                    set_node_default_value(slot.material, 1.0, material_input_type="Roughness")
                    set_node_default_value(slot.material, 0.0, material_input_type="Specular")
    bpy.ops.cats_armature.fix()
    time.sleep(0.1)
    bpy.ops.object.select_all(action='DESELECT')


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


def remove_alpha_image(object, remove_image: bool = False):
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
                                    if remove_image:
                                        bpy.data.images.remove(alpha_image)
                                node.inputs["Alpha"].default_value = 1.0


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(
        description='Import pmx file and convert to blend file.')
    parser.add_argument('--pmx_path', type=str,
                        help='path to pmx mesh')
    parser.add_argument('--output_folder', type=str,
                        help='path to exported mesh')
    args = parser.parse_args(raw_argv)

    pmx_path = args.pmx_path
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    obj_folder = os.path.join(output_folder, "OBJ")
    if not os.path.exists(obj_folder):
        os.mkdir(obj_folder)
    obj_path = os.path.join(obj_folder, "mesh.obj")
    fbx_folder = os.path.join(output_folder, "FBX")
    if not os.path.exists(fbx_folder):
        os.mkdir(fbx_folder)
    fbx_path = os.path.join(fbx_folder, "mesh.fbx")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    pmx_folder = os.path.split(pmx_path)[0]
    pmx_filename = os.path.split(pmx_path)[1]
    pmx_basename = os.path.splitext(pmx_filename)[0]
    blend_filename = pmx_basename + ".blend"
    blend_path = os.path.join(pmx_folder, blend_filename)

    pmx_import(pmx_path=pmx_path, blend_filepath=blend_path)

    for object_index, object_entity in enumerate(bpy.context.scene.objects):
        if object_entity.type == 'MESH':
            bpy.context.view_layer.objects.active = object_entity
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.separate(type='MATERIAL')
            bpy.ops.object.editmode_toggle()

    # meshes = []
    # for ind, obj in enumerate(bpy.context.scene.objects):
    #     if obj.type == 'MESH':
    #         remove_alpha_image(obj, remove_image=False)
    #         meshes.append(obj)
    bpy.ops.export_scene.fbx(filepath=fbx_path, path_mode='COPY')

    time.sleep(0.1)

    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_export(filepath=obj_path, path_mode='COPY')
    else:
        bpy.ops.export_scene.obj(filepath=obj_path, path_mode='COPY')

    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=False)
