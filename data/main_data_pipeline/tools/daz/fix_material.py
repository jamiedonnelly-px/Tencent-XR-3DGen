import argparse
import os
import sys
import time

import bpy


def load_mesh(mesh_path: str, z_up=False):
    bpy.ops.object.select_all(action='DESELECT')
    version_info = bpy.app.version
    print(mesh_path)
    print(version_info)
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


def toggle_alpha_linkage(object):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Alpha"].links) > 0:
                            l = node.inputs["Alpha"].links[0]
                            from_tex_node = l.from_node
                            node_tree.links.remove(l)
                            node_tree.links.new(
                                from_tex_node.outputs["Alpha"], node.inputs["Alpha"])


def fix_material_space(object, input_type="Metallic", color_space="Non-Color"):
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name
            print(material_name)
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(node.inputs)
                    if len(node.inputs[input_type].links) > 0:
                        l = node.inputs[input_type].links[0]
                        if l.from_socket.name == 'Color':
                            material_image_node = l.from_node
                            material_image_node.image.colorspace_settings.name = color_space


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, YZ_Axis=False):
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


def set_node_default_value(material, toset_value, material_input_type: str = "Base Color"):
    node_tree = material.node_tree
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs[material_input_type].default_value = toset_value


def check_mesh_with_certain_material(object, material_name: str = 'skin'):
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            if slot.material.name.lower() == material_name:
                return True
    return False


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Fix data material start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path to source mesh')
    parser.add_argument('--output_mesh_path', type=str, default="",
                        help='path to output mesh')
    parser.add_argument('--clear_pose', action='store_true',
                        help='clear default pose')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_path = args.source_mesh_path
    output_mesh_path = args.output_mesh_path
    clear_pose = args.clear_pose

    mesh_extension = os.path.splitext(source_mesh_path)[1]
    meshes = []
    if mesh_extension == ".fbx":
        try:
            bpy.ops.import_scene.fbx(filepath=source_mesh_path, use_anim=True)
        except:
            addon_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), '../../addons/better_fbx.zip')
            bpy.ops.preferences.addon_install(overwrite=True,
                                              target='DEFAULT',
                                              filepath=addon_path,
                                              filter_folder=True,
                                              filter_python=False,
                                              filter_glob="*.py;*.zip")
            bpy.ops.preferences.addon_enable(module="better_fbx")
            bpy.ops.better_import.fbx(filepath=source_mesh_path,
                                      use_optimize_for_blender=False,
                                      use_auto_bone_orientation=True,
                                      use_reset_mesh_origin=True,
                                      use_reset_mesh_rotation=True,
                                      use_detect_deform_bone=True,
                                      use_auto_smooth=True,
                                      use_animation=True)
        for ind, obj in enumerate(bpy.context.selected_objects):
            if obj.type == 'MESH':
                meshes.append(obj)
    elif mesh_extension == ".obj":
        meshes = load_mesh(source_mesh_path)

    print(meshes)
    if clear_pose and mesh_extension == ".fbx":
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.loc_clear()
        bpy.ops.pose.rot_clear()
        bpy.ops.pose.scale_clear()
        bpy.ops.object.posemode_toggle()

    for mesh in meshes:
        fix_material_space(mesh, input_type="Base Color", color_space='sRGB')
        toggle_alpha_linkage(mesh)

    for mesh in meshes:
        if mesh.material_slots:
            for slot in mesh.material_slots:
                node_tree = slot.material.node_tree
                material_name = slot.material.name
                set_node_default_value(
                    slot.material, 0.0, material_input_type='Metallic')
                set_node_default_value(
                    slot.material, 0.0, material_input_type='Specular')
                # set_node_default_value(
                #     slot.material, 0.0, material_input_type='Emission Strength')
                set_node_default_value(
                    slot.material, 0.5, material_input_type='Roughness')

    for image_data in bpy.data.images:
        image_name = image_data.name
        bpy.ops.image.unpack(method='WRITE_LOCAL', id=image_name)

    # for mesh in meshes:
    #     time.sleep(0.1)
    #     bpy.ops.object.select_all(action='DESELECT')
    #     bpy.context.view_layer.objects.active = mesh
    #     mesh.select_set(True)

    #     bpy.ops.object.mode_set(mode='EDIT')
    #     # Seperate by material
    #     bpy.ops.mesh.separate(type='MATERIAL')
    #     # Object Mode
    #     bpy.ops.object.mode_set(mode='OBJECT')

    # correct_mesh_list = []
    # to_delete_mesh_list = []
    # for ind, obj in enumerate(bpy.context.scene.objects):
    #     if obj.type == 'MESH':
    #         scalp_result = check_mesh_with_certain_material(
    #             obj, material_name='scalp')
    #         cornea_result = check_mesh_with_certain_material(
    #             obj, material_name='cornea')
    #         if scalp_result or cornea_result:
    #             to_delete_mesh_list.append(obj)
    #         else:
    #             correct_mesh_list.append(obj)

    # correct_mesh = join_list_of_mesh(correct_mesh_list)
    # for skin_mesh in to_delete_mesh_list:
    #     bpy.ops.object.select_all(action='DESELECT')
    #     bpy.context.view_layer.objects.active = skin_mesh
    #     skin_mesh.select_set(True)
    #     bpy.ops.object.delete(use_global=False)

    uncompressed_blend_path = output_mesh_path + ".blend"
    bpy.ops.wm.obj_export(filepath=output_mesh_path, path_mode='COPY')
    # bpy.ops.wm.save_as_mainfile(filepath=uncompressed_blend_path,
    #                             compress=False,
    #                             check_existing=False)
