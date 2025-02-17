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


def load_mesh(mesh_path: str):
    version_info = bpy.app.version
    if version_info[0] > 2:
        bpy.ops.wm.obj_import(filepath=mesh_path,
                              forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        bpy.ops.import_scene.obj(
            filepath=mesh_path, axis_forward='-Z', axis_up='Y')
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


def remesh_quad(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    bpy.context.object.data.remesh_mode = 'QUAD'
    bpy.ops.object.quadriflow_remesh()
    return the_mesh


def tris_quad(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    for vert in the_mesh.data.vertices:
        vert.select = True  # ensure all vertices are selected
    bpy.ops.object.mode_set(mode='EDIT')  # switch to edit mode
    bpy.ops.mesh.remove_doubles()  # remove doubles
    bpy.ops.mesh.tris_convert_to_quads()  # tris to quads
    bpy.ops.object.mode_set(mode='OBJECT')  # switch to object mode
    return the_mesh


def triangulate(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    the_mesh.modifiers.new("triangulate", "TRIANGULATE")
    bpy.ops.object.convert(target='MESH')  # bake modifier to mesh
    return the_mesh


def solidify(the_mesh, thickness=0.005):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    the_mesh.modifiers.new("solidify", "SOLIDIFY")
    the_mesh.modifiers["solidify"].thickness = thickness
    bpy.ops.object.convert(target='MESH')  # bake modifier to mesh
    return the_mesh


def decimate(the_mesh, target_faces_num):
    num_faces = len(the_mesh.data.polygons)
    if num_faces > target_faces_num:
        decimate_ratio = target_faces_num / num_faces
        decimator = the_mesh.modifiers.new("decimate", "DECIMATE")
        decimator.ratio = decimate_ratio
        bpy.ops.object.convert(target='MESH')
    return the_mesh


def exoside_quad(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    bpy.context.scene.qremesher.adaptive_size = 80
    bpy.context.scene.qremesher.target_count = 5000
    bpy.ops.qremesher.remesh()
    return the_mesh


def smart_uv(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    # lm = the_mesh.data.uv_layers.get("MyUV")
    # if not lm:
    #     lm = the_mesh.data.uv_layers.new(name="MyUV")
    # lm.active = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
    return the_mesh


def remesh(the_mesh, voxel_size):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    the_mesh.modifiers.new("remesh", "REMESH")
    the_mesh.modifiers["remesh"].voxel_size = voxel_size
    the_mesh.modifiers.new("triangulate", "TRIANGULATE")

    bpy.ops.object.convert(target='MESH')  # bake modifier to mesh
    return the_mesh


def create_material(object, material_name: str):
    mat = object.data.materials.get(material_name)

    if mat is None:
        mat = bpy.data.materials.new(material_name)
        object.data.materials.append(mat)

    mat.use_nodes = True

    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')

    links.new(shader.outputs[0], output.inputs[0])

    return mat


def change_texture_image(object, image_path: str, material_input_type: str = "Base Color", color_type: str = "sRGB"):
    if not os.path.exists(image_path):
        return
    texture_image = bpy.data.images.load(image_path)
    if object.material_slots:
        for slot in object.material_slots:
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

                    links.new(
                        diffusion_node.outputs["Color"], node.inputs[material_input_type])
                    if material_input_type == 'Base Color':
                        links.new(
                            diffusion_node.outputs["Alpha"], node.inputs["Alpha"])


def add_image_texture_node(object,
                           mesh_folder,
                           image_width: int = 128,
                           image_height: int = 128,
                           material_name: str = ""):
    if object.material_slots:
        for i in range(len(object.material_slots)):
            with bpy.context.temp_override(object=object):
                bpy.ops.object.material_slot_remove()

    material_name = material_name.replace(" ", "_")
    create_material(object=object, material_name=material_name)
    diffuse_image = bpy.data.images.new(material_name + "_d", width=image_width, height=image_height)
    image_texture_name = material_name + "_wsb_d.png"
    image_texture_path = os.path.join(mesh_folder, image_texture_name)
    diffuse_image.save_render(image_texture_path)

    change_texture_image(the_mesh, image_texture_path)

    return the_mesh


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, z_up=False):
    print("export mesh", mesh, "# polygons", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    version_info = bpy.app.version
    if version_info[0] > 2:
        if z_up:
            bpy.ops.wm.obj_export(filepath=mesh_path,
                                  path_mode=path_mode,
                                  forward_axis='Y', up_axis='Z',
                                  global_scale=global_scale,
                                  export_selected_objects=True)
        else:
            bpy.ops.wm.obj_export(filepath=mesh_path,
                                  path_mode=path_mode,
                                  forward_axis='NEGATIVE_Z', up_axis='Y',
                                  global_scale=global_scale,
                                  export_selected_objects=True)
    else:
        if z_up:
            bpy.ops.export_scene.obj(filepath=mesh_path,
                                     use_selection=True,
                                     path_mode=path_mode,
                                     axis_forward='Y', axis_up='Z',
                                     global_scale=global_scale)
        else:
            bpy.ops.export_scene.obj(filepath=mesh_path,
                                     use_selection=True,
                                     path_mode=path_mode,
                                     axis_forward='-Z', axis_up='Y',
                                     global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Remesh op start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--output_mesh_path', type=str,
                        help='path of remesh processed mesh')
    parser.add_argument('--remesh_voxel_size', type=float, default=0.002,
                        help='triangulation voxel size')
    parser.add_argument('--solidify_thickness', type=float, default=0.01,
                        help='thickness of solidify')
    parser.add_argument('--decimate_faces_num', type=int, default=50000,
                        help='face number used in mesh simplification')
    parser.add_argument('--process_stages', type=str, default="decimate+quad",
                        help='stages used in remesh process')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_path = args.mesh_path
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_folder = os.path.split(mesh_path)[0]
    output_mesh_path = args.output_mesh_path
    remesh_voxel_size = args.remesh_voxel_size
    solidify_thickness = args.solidify_thickness
    decimate_faces_num = args.decimate_faces_num

    process_stages_str = args.process_stages
    process_stages = process_stages_str.split('+')
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_name = os.path.split(mesh_path)[1]

    stage_function_map = {}
    stage_function_map['exoside'] = exoside_quad
    stage_function_map['to_quad'] = tris_quad
    stage_function_map['quad'] = remesh_quad
    stage_function_map['remesh'] = remesh
    stage_function_map['decimate'] = decimate
    stage_function_map['solidify'] = solidify
    stage_function_map['triangulate'] = triangulate
    stage_function_map['smart_uv'] = smart_uv
    stage_function_map['add_image'] = add_image_texture_node

    manifold_meshes = load_mesh(mesh_path)
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            manifold_meshes.append(obj)
    the_mesh = join_list_of_mesh(manifold_meshes)

    for stage_index in range(len(process_stages)):
        stage_name = process_stages[stage_index]
        if stage_name not in stage_function_map.keys():
            print("Cannot find processing stage %s in stage maps")
            continue
        print("Stage #%i is %s; input mesh has %i polygons.........." %
              (stage_index, stage_name, len(the_mesh.data.polygons)))
        if stage_name == 'decimate':
            the_mesh = stage_function_map[stage_name](the_mesh, decimate_faces_num)
        elif stage_name == 'solidify':
            the_mesh = stage_function_map[stage_name](the_mesh, solidify_thickness)
        elif stage_name == 'remesh':
            the_mesh = stage_function_map[stage_name](the_mesh, remesh_voxel_size)
        elif stage_name == 'add_image':
            the_mesh = stage_function_map[stage_name](the_mesh, mesh_folder, material_name=mesh_basename)
        else:
            the_mesh = stage_function_map[stage_name](the_mesh)
        print("# polygons", len(the_mesh.data.polygons))

    print("Finish geometry processing.......")
    export_mesh_obj(the_mesh, output_mesh_path, 'COPY', z_up=False)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Remesh op done. Local time is %s" % (local_time_str))