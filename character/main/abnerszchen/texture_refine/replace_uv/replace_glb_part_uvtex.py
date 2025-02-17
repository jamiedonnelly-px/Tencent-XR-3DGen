import bpy
import bmesh
import os
import time
import json
import math
import argparse
import random
import sys
import gc
import numpy as np


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

def change_texture_image(object,
                         image_path: str,
                         material_name: str = "",
                         material_input_type: str = "Base Color",
                         color_type: str = "sRGB"):
    if not os.path.exists(image_path):
        return
    texture_image = bpy.data.images.load(image_path)
    if object.material_slots:
        for slot in object.material_slots:
            if len(material_name) > 0:
                if slot.material.name != material_name:
                    continue
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            diffusion_node = nodes.new("ShaderNodeTexImage")
            diffusion_node.image = texture_image
            diffusion_node.image.colorspace_settings.name = color_type

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(node.inputs)
                    if len(node.inputs[material_input_type].links) > 0:
                        l = node.inputs[material_input_type].links[0]
                        original_tex_image_node = l.from_node
                        if l is not None:
                            node_tree.links.remove(l)
                        if original_tex_image_node is not None:
                            nodes.remove(original_tex_image_node)

                    links.new(diffusion_node.outputs["Color"], node.inputs[material_input_type])
                    if material_input_type == 'Base Color':
                        links.new(diffusion_node.outputs["Alpha"], node.inputs["Alpha"])

def parse_input_img_name_pairs(input_image_paths, object_part_names):
    if not input_image_paths or not object_part_names:
        print(f"[ERROR] empty input: ", input_image_paths, object_part_names)
        return False, {}
    if len(input_image_paths) != len(object_part_names):
        print(f"[ERROR] input {len(input_image_paths)} != {len(object_part_names)}", input_image_paths, object_part_names)
        return False, {}
    
    input_name_img_dict = {}
    for input_image_path, object_part_name in zip(input_image_paths, object_part_names):
        if os.path.exists(input_image_path):
            input_name_img_dict[object_part_name] = input_image_path
        else:
            print(f"[ERROR] can not find input_image_path {input_image_path}")
            
    input_flag = len(input_name_img_dict) >= 1
    print('input_name_img_dict=', input_name_img_dict)            
    return input_flag, input_name_img_dict

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def call_replace_glb_once(source_mesh_path, input_name_img_dict, output_mesh_path, debug_blend_save=False):
    ts = time.time()
    clear_scene()
    source_mesh_extension = os.path.splitext(source_mesh_path)[1].lower()
    output_mesh_folder=os.path.split(output_mesh_path)[0]
    os.makedirs(output_mesh_folder, exist_ok=True)

    source_mesh_list = []
    if source_mesh_extension == ".glb":
        bpy.ops.import_scene.gltf(filepath=source_mesh_path)
        bpy.ops.object.select_all(action='DESELECT')
        for ind, obj in enumerate(bpy.context.scene.objects):
            if obj.type == 'MESH':
                source_mesh_list.append(obj)
    elif source_mesh_extension==".obj":
        source_mesh_list = load_mesh(source_mesh_path)
    
    flag = False
    for mesh in source_mesh_list:
        if mesh.name in input_name_img_dict:
            change_texture_image(mesh, input_name_img_dict[mesh.name])
            flag = True

    if flag:
        if debug_blend_save:
            name = os.path.splitext(source_mesh_path)[0]
            debug_blend_file = os.path.join(output_mesh_folder, f"{name}_debug.blend")
            bpy.ops.file.pack_all()
            bpy.ops.wm.save_as_mainfile(filepath=debug_blend_file,
                                        compress=False,
                                        check_existing=False)        
        bpy.ops.export_scene.gltf(filepath=output_mesh_path)
    else:
        print(f"[ERROR] replace_glb failed. name not match!!!input_name_img_dict = ", input_name_img_dict)
        for mesh in source_mesh_list:
            print('[ERROR] but mesh.name = ', mesh.name)
        
    tuse = time.time() - ts
    print('tuse: ', tuse)
    return

if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path to source mesh')
    parser.add_argument('--input_image_paths', type=str, nargs='+', required=True,
                        help='path to input image that you wish to substitute')
    parser.add_argument('--object_part_names', type=str, nargs='+', required=True,
                        help='name of object part that needs to be substituted')
    parser.add_argument('--output_mesh_path', type=str,
                        help='path to output mesh, with uv from source and geometry from destination')
    parser.add_argument('--debug_blend_save', action='store_true',
                        help='save blend file in render process for debug...')
    args = parser.parse_args(raw_argv)

    ts = time.time()
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_path = args.source_mesh_path
    input_image_paths = args.input_image_paths
    object_part_names = args.object_part_names
    output_mesh_path = args.output_mesh_path
    debug_blend_save = args.debug_blend_save

    input_flag, input_name_img_dict = parse_input_img_name_pairs(input_image_paths, object_part_names)
    if not input_flag:
        print('[ERROR] parse_input_img_name_pairs failed. exit')
        exit()
        
    call_replace_glb_once(source_mesh_path, input_name_img_dict, output_mesh_path, debug_blend_save=debug_blend_save)
    
    source_animation_glb = os.path.splitext(source_mesh_path)[0] + "_animation.glb"
    if os.path.exists(source_animation_glb):
        print(f'source_animation_glb = {source_animation_glb}')
        output_animation_path = os.path.splitext(output_mesh_path)[0] + "_animation.glb"
        call_replace_glb_once(source_animation_glb, input_name_img_dict, output_animation_path, debug_blend_save=debug_blend_save)
        
        
    tuse = time.time() - ts
    print('tuse all ', tuse)