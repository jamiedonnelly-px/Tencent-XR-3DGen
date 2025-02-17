import argparse
import os
import re
import sys
import time

import bpy

weapon = ["weapon", "Weapon"]


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


def pmx_process():
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


def check_blend_material_status(the_mesh_list):
    for the_mesh in the_mesh_list:
        if the_mesh.material_slots:
            for slot in the_mesh.material_slots:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    print(node.name)
                    if node.name == "Principled BSDF" or node.name == '原理化BSDF':
                        return True
    return False


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(
        description='Import pmx file and convert to blend file.')
    parser.add_argument('--pmx_blend_path', type=str,
                        help='path to pmx mesh')
    parser.add_argument('--output_folder', type=str,
                        help='path to exported mesh')
    args = parser.parse_args(raw_argv)

    pmx_blend_path = args.pmx_blend_path
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    texture_folder = os.path.join(output_folder, "texture")
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

    bpy.ops.wm.open_mainfile(filepath=pmx_blend_path, load_ui=False)
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
        elif obj.type == 'LIGHT':
            obj.select_set(state=True)
            bpy.ops.object.delete()
        elif obj.type == 'CAMERA':
            obj.select_set(state=True)
            bpy.ops.object.delete()

    if check_blend_material_status(meshes):
        joint_mesh = join_list_of_mesh(meshes)
        bpy.context.scene.view_settings.view_transform = 'Standard'
        pattern = re.compile(r"(\.\d+)$")
        image_name_block_map = {}
        image_save_path = {}
        if joint_mesh.material_slots:
            for slot in joint_mesh.material_slots:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.name == "mmd_base_tex":
                        image_basename = pattern.sub("", node.image.name)
                        if image_basename not in image_name_block_map.keys():
                            image_name_block_map[image_basename] = node.image
                        else:
                            node.image = image_name_block_map[image_basename]

                for node in node_tree.nodes:
                    if node.name == "mmd_base_tex":
                        if node.image.name not in image_save_path.keys():
                            image_path = os.path.join(texture_folder, node.image.name + ".png")
                            node.image.save_render(image_path)
                            image_save_path[node.image.name] = image_path
                            texture_image = bpy.data.images.load(image_path)
                            node.image = texture_image
                            node.image.colorspace_settings.name = 'sRGB'
                        else:
                            texture_image = bpy.data.images.load(image_save_path[node.image.name])
                            node.image = texture_image
                            node.image.colorspace_settings.name = 'sRGB'
    else:
        pmx_process()

    bpy.ops.export_scene.fbx(filepath=fbx_path, path_mode='COPY')
    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_export(filepath=obj_path, path_mode='COPY')
    else:
        bpy.ops.export_scene.obj(filepath=obj_path, path_mode='COPY')
