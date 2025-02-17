import argparse
import json
import os
import sys
import time

import bpy
import numpy as np


def load_mesh(mesh_path: str, XY_Axis=False):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]
    mesh_extension = mesh_extension.lower()
    if mesh_extension == ".obj":
        version_info = bpy.app.version
        if version_info[0] > 2:
            bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis='NEGATIVE_Z', up_axis='Y')
        else:
            bpy.ops.import_scene.obj(filepath=mesh_path, axis_forward='-Z', axis_up='Y')
        bpy.ops.object.select_all(action='DESELECT')
        meshes = []
        for ind, obj in enumerate(bpy.context.scene.objects):
            if obj.type == 'MESH':
                meshes.append(obj)
        return meshes
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


def check_texture(the_mesh, material_input_type="Base Color"):
    version_info = bpy.app.version
    print(version_info)
    if version_info[0] >= 4:
        if material_input_type == "Specular":
            material_input_type = "Specular IOR Level"
    if the_mesh.material_slots:
        for slot in the_mesh.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs[material_input_type].links) > 0:
                            return True
    return False


def check_normal(the_mesh):
    if the_mesh.material_slots:
        for slot in the_mesh.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Normal"].links) > 0:
                            normal_map_node = node.inputs["Normal"].links[0].from_node
                            if len(normal_map_node.inputs["Color"].links) > 0:
                                return True
    return False


def check_face_number(the_mesh):
    return len(the_mesh.data.polygons)


def check_white_model(the_mesh):
    bsdf_material_number = 0
    material_number = 0

    if the_mesh.material_slots:
        for slot in the_mesh.material_slots:
            material_number = material_number + 1
            material_name = slot.material.name
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_material_number = bsdf_material_number + 1

    if bsdf_material_number < int(0.1 * material_number):
        exit(-1)

    white_model = True

    color_value_list = []

    if the_mesh.material_slots:
        for slot in the_mesh.material_slots:
            material_name = slot.material.name
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if len(node.inputs["Base Color"].links) > 0:
                        white_model = False
                    else:
                        color_value = np.array([node.inputs["Base Color"].default_value[0],
                                                node.inputs["Base Color"].default_value[1],
                                                node.inputs["Base Color"].default_value[2],
                                                node.inputs["Base Color"].default_value[3]])
                        if len(color_value_list) == 0:
                            color_value_list.append(color_value)
                        else:
                            for other_value in color_value_list:
                                check_same_array = (other_value == color_value)
                                if check_same_array.all():
                                    break
                                color_value_list.append(color_value)
    print(len(color_value_list))
    if len(color_value_list) > 3:
        white_model = False
    return white_model


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Parse mesh info start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be parsed')
    parser.add_argument('--output_mesh_json_info', type=str, default="",
                        help='path of manifold processed mesh')
    parser.add_argument('--info_stage_str', type=str, default="",
                        help='stages to be classified')

    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_path = args.mesh_path
    output_mesh_json_info = args.output_mesh_json_info
    info_stage_str = args.info_stage_str
    info_stages = info_stage_str.split("+")

    manifold_meshes = load_mesh(mesh_path)
    mesh_list = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            mesh_list.append(obj)
    if len(mesh_list) > 0:
        the_mesh = join_list_of_mesh(mesh_list)

        mesh_info = {}

        for stage_name in info_stages:
            if stage_name == "TextureExist":
                mesh_info["TextureExist"] = check_texture(the_mesh=the_mesh, material_input_type='Base Color')
            elif stage_name == 'RoughnessExist':
                mesh_info["RoughnessExist"] = check_texture(the_mesh=the_mesh, material_input_type='Roughness')
            elif stage_name == 'MetallicExist':
                mesh_info["MetallicExist"] = check_texture(the_mesh=the_mesh, material_input_type='Metallic')
            elif stage_name == 'SpecularExist':
                mesh_info["SpecularExist"] = check_texture(the_mesh=the_mesh, material_input_type='Specular')
            elif stage_name == 'NormalExist':
                mesh_info["NormalExist"] = check_normal(the_mesh=the_mesh)
            elif stage_name == 'FaceNum':
                mesh_info["FaceNum"] = check_face_number(the_mesh=the_mesh)
            else:
                mesh_info["White"] = check_white_model(the_mesh=the_mesh)

        if len(output_mesh_json_info) > 0:
            write_json(output_mesh_json_info, mesh_info)

        print("finish parse mesh at %s" % (mesh_path))
    else:
        print("cannot load mesh into blender....mesh list length is 0")
