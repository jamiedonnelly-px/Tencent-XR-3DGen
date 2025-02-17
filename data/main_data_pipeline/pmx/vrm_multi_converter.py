import argparse
import gc
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


def remove_image_linkage(object, material_input_type: str = "Roughness",
                         remove_tex_image: bool = True):
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if version_info[0] >= 4:
                        if material_input_type == "Specular":
                            material_input_type = "Specular IOR Level"
                    if len(node.inputs[material_input_type].links) > 0:
                        l = node.inputs[material_input_type].links[0]
                        original_tex_image_node = l.from_node
                        if l is not None:
                            links.remove(l)
                        if remove_tex_image:
                            if original_tex_image_node is not None:
                                nodes.remove(original_tex_image_node)
                    if isinstance(node.inputs[material_input_type].default_value, float):
                        node.inputs[material_input_type].default_value = 0
                    else:
                        node.inputs[material_input_type].default_value = (0, 0, 0, 1)


def set_node_default_value(material, toset_value, material_input_type: str = "Base Color"):
    version_info = bpy.app.version
    node_tree = material.node_tree
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            if version_info[0] >= 4:
                if material_input_type == "Specular":
                    material_input_type = "Specular IOR Level"
            node.inputs[material_input_type].default_value = toset_value


def set_default_value(object, toset_value: float, material_input_type: str = "Metallic"):
    if object.material_slots:
        for slot in object.material_slots:
            set_node_default_value(
                slot.material, toset_value, material_input_type)


def remove_node_by_name(object, node_name: str):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.name == node_name:
                        node_tree.nodes.remove(node)


def toggle_alpha_blend_mode(object, blend_method='OPAQUE'):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                slot.material.blend_method = blend_method


def vrm_import(vrm_path: str, blend_filepath: str):
    bpy.ops.wm.vrm_license_warning(filepath=vrm_path,
                                   license_confirmations=[{"name": "LicenseConfirmation0",
                                                           "message": "This VRM is licensed by VRoid Hub License \"Alterations: No\".",
                                                           "url": "https://hub.vroid.com/license?allowed_to_use_user=everyone&characterization_allowed_user=author&corporate_commercial_use=disallow&credit=necessary&modification=disallow&personal_commercial_use=disallow&redistribution=disallow&sexual_expression=disallow&version=1&violent_expression=disallow",
                                                           "json_key": "otherPermissionUrl"},
                                                          {"name": "LicenseConfirmation1",
                                                           "message": "This VRM is licensed by VRoid Hub License \"Alterations: No\".",
                                                           "url": "https://hub.vroid.com/license?allowed_to_use_user=everyone&characterization_allowed_user=author&corporate_commercial_use=disallow&credit=necessary&modification=disallow&personal_commercial_use=disallow&redistribution=disallow&sexual_expression=disallow&version=1&violent_expression=disallow",
                                                           "json_key": "otherLicenseUrl"}],
                                   import_anyway=True,
                                   extract_textures_into_folder=False,
                                   make_new_texture_folder=True)
    time.sleep(0.1)
    bpy.ops.wm.save_as_mainfile(filepath=blend_filepath, compress=False, check_existing=False)
    # bpy.ops.object.select_all(action='DESELECT')
    # meshes = []
    # for ind, obj in enumerate(bpy.context.scene.objects):
    #     if obj.type == 'MESH':
    #         meshes.append(obj)
    #
    # textured_mesh = join_list_of_mesh(meshes)
    bpy.ops.file.unpack_all(method='WRITE_LOCAL')
    bpy.ops.wm.save_as_mainfile(filepath=blend_filepath, compress=False, check_existing=False)


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


def remove_metallic(the_mesh):
    # remove_image_linkage(the_mesh, material_input_type="Metallic")
    set_default_value(the_mesh, 0.0, material_input_type="Metallic")
    return the_mesh


def remove_roughness(the_mesh):
    # remove_image_linkage(the_mesh, material_input_type="Metallic")
    set_default_value(the_mesh, 0.3, material_input_type="Roughness")
    return the_mesh


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(
        description='Import pmx file and convert to blend file.')
    parser.add_argument('--vrm_path', type=str,
                        help='path to pmx mesh')
    parser.add_argument('--output_folder', type=str,
                        help='path to exported mesh')
    args = parser.parse_args(raw_argv)

    vrm_path = args.vrm_path
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    blend_folder = os.path.join(output_folder, "blend")
    if not os.path.exists(blend_folder):
        os.mkdir(blend_folder)
    vrm_filename = os.path.split(vrm_path)[1]
    vrm_basename = os.path.splitext(vrm_filename)[0]
    blend_filename = vrm_basename + ".blend"
    blend_fbx_filename = vrm_basename + ".fbx"
    blend_obj_filename = vrm_basename + ".obj"
    blend_folder_blend_filepath = os.path.join(blend_folder, blend_filename)
    blend_folder_fbx_filepath = os.path.join(blend_folder, blend_fbx_filename)
    blend_folder_obj_filepath = os.path.join(blend_folder, blend_obj_filename)

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

    vrm_import(vrm_path=vrm_path, blend_filepath=blend_folder_blend_filepath)
    bpy.ops.export_scene.fbx(filepath=blend_folder_fbx_filepath, path_mode='AUTO')

    time.sleep(0.1)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # Remove unused (orphan) data-blocks
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    # Clean up memory and force garbage collection
    bpy.ops.outliner.orphans_purge()
    gc.collect()

    time.sleep(0.1)

    bpy.ops.import_scene.fbx(filepath=blend_folder_fbx_filepath, use_anim=True)
    meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    joint_mesh = join_list_of_mesh(meshes)

    remove_metallic(joint_mesh)
    remove_roughness(joint_mesh)
    toggle_alpha_blend_mode(joint_mesh, blend_method='CLIP')

    for object_index, object_entity in enumerate(bpy.context.scene.objects):
        if object_entity.type == 'MESH':
            bpy.context.view_layer.objects.active = object_entity
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.separate(type='MATERIAL')
            bpy.ops.object.editmode_toggle()

    bpy.ops.export_scene.fbx(filepath=fbx_path, path_mode='COPY')

    time.sleep(0.1)

    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_export(filepath=obj_path, path_mode='COPY')
    else:
        bpy.ops.export_scene.obj(filepath=obj_path, path_mode='COPY')
