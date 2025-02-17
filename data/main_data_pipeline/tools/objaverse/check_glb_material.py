import argparse
import json
import sys
import time

import bpy


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def fill_default_value_for_material(material_stat_map: dict):
    material_stat_map["roughness"] = {}
    material_stat_map["roughness"]["value"] = False
    material_stat_map["roughness"]["image"] = False

    material_stat_map["metallic"] = {}
    material_stat_map["metallic"]["value"] = False
    material_stat_map["metallic"]["image"] = False

    material_stat_map["specular"] = {}
    material_stat_map["specular"]["value"] = False
    material_stat_map["specular"]["image"] = False


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
    parser.add_argument('--output_json_path', type=str, default="",
                        help='path to output mesh info json')
    parser.add_argument('--verify_material_number', action='store_true',
                        help='add objects with multiple material into this ')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_path = args.source_mesh_path
    output_json_path = args.output_json_path
    verify_material_number = args.verify_material_number

    bpy.ops.import_scene.gltf(filepath=source_mesh_path)

    meshes = []
    for ind, obj in enumerate(bpy.context.selected_objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    material_stat_map = {}
    bsdf_material_number = 0
    material_number = 0
    for mesh in meshes:
        if mesh.material_slots:
            for slot in mesh.material_slots:
                material_number = material_number + 1
                material_name = slot.material.name
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        bsdf_material_number = bsdf_material_number + 1

    if bsdf_material_number < int(0.1 * material_number):
        exit(-1)

    fill_default_value_for_material(material_stat_map=material_stat_map)

    roughness_value_set = set()
    metallic_value_set = set()
    specular_value_set = set()
    for mesh in meshes:
        if mesh.material_slots:
            for slot in mesh.material_slots:
                material_name = slot.material.name
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Roughness"].links) > 0:
                            material_stat_map["roughness"]["image"] = True
                        else:
                            if verify_material_number:
                                if node.inputs["Roughness"].default_value > 0:
                                    roughness_value_set.add(
                                        node.inputs["Roughness"].default_value)
                                    if node.inputs["Roughness"].default_value not in roughness_value_set:
                                        if len(roughness_value_set) > 2:
                                            material_stat_map["roughness"]["value"] = True

                        if len(node.inputs["Metallic"].links) > 0:
                            material_stat_map["metallic"]["image"] = True
                        else:
                            if verify_material_number:
                                if node.inputs["Metallic"].default_value > 0:
                                    metallic_value_set.add(
                                        node.inputs["Metallic"].default_value)
                                    if node.inputs["Metallic"].default_value not in metallic_value_set:
                                        if len(metallic_value_set) > 2:
                                            material_stat_map["metallic"]["value"] = True

                        if len(node.inputs["Specular"].links) > 0:
                            material_stat_map["specular"]["image"] = True
                        else:
                            if verify_material_number:
                                if node.inputs["Specular"].default_value > 0:
                                    specular_value_set.add(
                                        node.inputs["Specular"].default_value)
                                    if node.inputs["Specular"].default_value not in specular_value_set:
                                        if len(specular_value_set) > 2:
                                            material_stat_map["specular"]["value"] = True

    if not verify_material_number:
        for socket_name in material_stat_map.keys():
            if material_stat_map[socket_name]["image"]:
                write_json(output_json_path, material_stat_map)
                break
    else:
        for socket_name in material_stat_map.keys():
            if material_stat_map[socket_name]["image"] or material_stat_map[socket_name]["value"]:
                write_json(output_json_path, material_stat_map)
                break
