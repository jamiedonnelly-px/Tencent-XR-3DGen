# -*- coding: utf-8 -*-
# run this script in blender's python

import argparse
import gc
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time

import bmesh
import bpy
import miniball
import numpy as np
import trimesh
from mathutils import Matrix, Vector
from numpy import arange, sin, cos, arccos

logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')


def run_cmd(cmd_str: str, slient: bool = False):
    """
    Run a shell command using subprocess
    :param cmd_str: shell command
    :return: True if success, False otherwise
    """
    cmd_elements = shlex.split(cmd_str)
    try:
        if slient:
            subprocess.run(cmd_elements, check=True, text=True, stdout=subprocess.DEVNULL, )
        else:
            subprocess.run(cmd_elements, check=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error("Error in running cmd %s" % cmd_str)
        logging.error("Error code %s" % e.returncode)
        logging.error("Error msg %s" % e.stderr)
        return False
    return True


def add_area_light(location,
                   rotation,
                   energy: float = 50.0,
                   area_shape: str = "Square",
                   size: float = 5.0,
                   size_y: float = 5.0,
                   light_name: str = "main"):
    """
    Add a blender area light to the scene
    :param location: location of the area light in the scene
    :param rotation: rotation of the area light in the scene
    :param energy: energy of the area light
    :param area_shape: shape of the area light, can be "Rectangle" or "Square"
    :param size: size of the area light, x size when area_shape is "Rectangle"
    :param size_y: y size of the area light, only used when area_shape is "Rectangle"
    :param light_name: name of the light
    """
    light_data2 = bpy.data.lights.new(name=light_name + "_data", type='AREA')
    light_data2.energy = energy
    if area_shape == "Rectangle":
        bpy.context.object.data.shape = 'RECTANGLE'
        light_data2.size_y = size_y
    else:
        bpy.context.object.data.shape = 'SQUARE'

    light_data2.size = size
    light_object2 = bpy.data.objects.new(name=light_name + "_light", object_data=light_data2)
    light_object2.location = location
    light_object2.rotation_euler = rotation
    bpy.context.collection.objects.link(light_object2)


def add_sun_light(location,
                  energy: float = 5.0,
                  specular_factor: float = 1.0,
                  use_shadow: bool = False,
                  light_name: str = "sun"):
    """
    Add a blender sun light to the scene
    :param location: location of the sun in the scene
    :param energy: energy of the sun light
    :param specular_factor: specular factor of the sun light
    :param use_shadow: if true, use shadow in sun light
    :param light_name: name of the light
    """
    light_data2 = bpy.data.lights.new(name=light_name + "_data", type='SUN')
    light_data2.energy = energy
    light_data2.use_shadow = use_shadow
    # disable specular shading if desired; some object may render badly with specular
    light_data2.specular_factor = specular_factor
    light_object2 = bpy.data.objects.new(name=light_name + "_light", object_data=light_data2)
    light_object2.location = location
    bpy.context.collection.objects.link(light_object2)


def remove_image_linkage(object, material_input_type: str = "Roughness",
                         remove_tex_image: bool = True):
    """
    Remove the image linkage and the tex_image node (if needed) from the object's material
    :param object: blender object containing the material to be converted
    :param material_input_type: the name of the input slot on BSDF node that needs to be removed
    :param remove_tex_image: if true, remove the tex_image node as well
    """
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # blender changes BSDF specular api after version 4.0
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


def clear_blender_world():
    """
    Clear all the nodes in the current blender world.
    This is necessary to avoid the influence of default light in some rendering mode.
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes

    # remove current nodes as we need to reduce effects of other output heads
    for node in nodes:
        nodes.remove(node)


def change_hdr_map_path(hdr_map_path: str, strength: float):
    """
    Change current hdr map in blender world to the given path.
    :param hdr_map_path: path to the hdr map, should be an .hdr file, for instance: https://polyhaven.com/hdris
    """
    hdr_file = hdr_map_path
    if len(hdr_file) < 1 or not os.path.exists(hdr_file):
        hdr_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'irrmaps/aerodynamics_workshop_2k.hdr')
    hdr_image = bpy.data.images.load(hdr_file)

    # setup scene (world) texture
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # remove current nodes as we need to reduce effects of other output heads
    for node in nodes:
        nodes.remove(node)

    # create new nodes for env map
    environment_texture_node = nodes.new("ShaderNodeTexEnvironment")
    background_node = nodes.new("ShaderNodeBackground")
    background_node.inputs["Strength"].default_value = strength
    environment_texture_node.image = hdr_image
    output_node = nodes.new("ShaderNodeOutputWorld")

    # connect the nodes
    links.new(environment_texture_node.outputs["Color"], background_node.inputs["Color"])
    links.new(background_node.outputs["Background"], output_node.inputs["Surface"])
    bpy.context.scene.render.film_transparent = True


def fix_bump_color_space(object):
    """
    Fix bump map color space to non-color space to avoid color shift.
    :param object: blender object containing the material to be converted
    """
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
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
    """
    Fix input material image color space to non-color space to avoid color shift.
    :param object: blender object containing the material to be converted
    :param input_type: input type of the bump map, can be 'Metallic', 'Roughness', 'Specular'
    """
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # blender changes BSDF specular api after version 4.0
                    if version_info[0] >= 4:
                        if input_type == "Specular":
                            input_type = "Specular IOR Level"
                    if len(node.inputs[input_type].links) > 0:
                        l = node.inputs[input_type].links[0]
                        if l.from_socket.name == 'Color':
                            material_image_node = l.from_node
                            material_image_node.image.colorspace_settings.name = "Non-Color"


def pbr_render_shader(object):
    """
    Generate rendering shader for roughness and metallic;
    rendered image has color [R (always 1.0), G (roughness), B (metallic)].
    :param object: blender object containing the material to be converted
    """
    version_info = bpy.app.version
    remove_image_linkage(object, material_input_type='Base Color', remove_tex_image=False)
    # blender changed BSDF emission api after version 4.0
    if version_info[0] > 3:
        remove_image_linkage(object, material_input_type='Emission Color', remove_tex_image=False)
    else:
        remove_image_linkage(object, material_input_type='Emission', remove_tex_image=False)

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            newly_made_bsdf_list = []
            for node in node_tree.nodes:
                if "BSDF" in node.type:
                    if node in newly_made_bsdf_list:
                        continue
                    new_bsdf_shader = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
                    newly_made_bsdf_list.append(new_bsdf_shader)

                    if not node.outputs["BSDF"].is_linked:
                        continue
                    output_link = node.outputs["BSDF"].links[0]
                    output_socket = output_link.to_socket
                    node_tree.links.remove(output_link)
                    node_tree.links.new(new_bsdf_shader.outputs["BSDF"], output_socket)

                    # follow the common standard to use one RGB image as material image
                    combine_node = node_tree.nodes.new(type="ShaderNodeCombineColor")
                    combine_node.inputs['Red'].default_value = 1.0
                    if "Roughness" in node.inputs:
                        if node.inputs["Roughness"].is_linked:
                            color_origin_socket = node.inputs["Roughness"].links[0].from_socket
                            node_tree.links.new(color_origin_socket, combine_node.inputs['Green'])
                        else:
                            combine_node.inputs['Green'].default_value = node.inputs["Roughness"].default_value
                    if "Metallic" in node.inputs:
                        if node.inputs["Metallic"].is_linked:
                            color_origin_socket = node.inputs["Metallic"].links[0].from_socket
                            node_tree.links.new(color_origin_socket, combine_node.inputs['Blue'])
                        else:
                            combine_node.inputs['Blue'].default_value = node.inputs["Metallic"].default_value

                    new_bsdf_shader.inputs["Base Color"].default_value = (0, 0, 0, 1)
                    new_bsdf_shader.inputs["Emission Strength"].default_value = 1
                    if version_info[0] > 3:
                        node_tree.links.new(combine_node.outputs["Color"], new_bsdf_shader.inputs["Emission Color"])
                    else:
                        node_tree.links.new(combine_node.outputs["Color"], new_bsdf_shader.inputs["Emission"])

                    if "Alpha" in node.inputs:
                        if node.inputs["Alpha"].is_linked:
                            color_origin_socket = node.inputs["Alpha"].links[0].from_socket
                            node_tree.links.new(color_origin_socket, new_bsdf_shader.inputs['Alpha'])
                        else:
                            new_bsdf_shader.inputs['Alpha'].default_value = node.inputs["Alpha"].default_value


def bump_render_shader(object):
    """
    generate rendering shader for bump map
    :param object: blender object containing the material to be converted
    """
    version_info = bpy.app.version
    remove_image_linkage(object, material_input_type='Base Color', remove_tex_image=False)
    if version_info[0] > 3:
        remove_image_linkage(object, material_input_type='Emission Color', remove_tex_image=False)
    else:
        remove_image_linkage(object, material_input_type='Emission', remove_tex_image=False)

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            newly_made_bsdf_list = []
            for node in node_tree.nodes:
                if "BSDF" in node.type:
                    if node in newly_made_bsdf_list:
                        continue
                    new_bsdf_shader = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
                    newly_made_bsdf_list.append(new_bsdf_shader)

                    if not node.outputs["BSDF"].is_linked:
                        continue
                    output_link = node.outputs["BSDF"].links[0]
                    output_socket = output_link.to_socket
                    node_tree.links.remove(output_link)
                    node_tree.links.new(new_bsdf_shader.outputs["BSDF"], output_socket)

                    new_bsdf_shader.inputs["Base Color"].default_value = (0, 0, 0, 1)
                    new_bsdf_shader.inputs["Emission Strength"].default_value = 1

                    if "Normal" in node.inputs:
                        if node.inputs['Normal'].is_linked:
                            normal_map_node = node.inputs['Normal'].links[0].from_node
                            if normal_map_node.type == 'NORMAL_MAP' and normal_map_node.space == 'TANGENT':
                                bump_texture_node = normal_map_node.inputs['Color'].links[0].from_node
                                if bump_texture_node.type == 'TEX_IMAGE':
                                    if version_info[0] > 3:
                                        node_tree.links.new(bump_texture_node.outputs["Color"],
                                                            new_bsdf_shader.inputs["Emission Color"])
                                    else:
                                        node_tree.links.new(bump_texture_node.outputs["Color"],
                                                            new_bsdf_shader.inputs["Emission"])
                                # default normal value is from
                                # https://github.com/3DTopia/MaterialAnything/blob/main/rendering_scripts/blender_script_material.py
                                else:
                                    if version_info[0] > 3:
                                        new_bsdf_shader.inputs["Emission Color"].default_value = (0.5, 0.5, 1, 1)
                                    else:
                                        new_bsdf_shader.inputs["Emission"].default_value = (0.5, 0.5, 1, 1)
                            else:
                                if version_info[0] > 3:
                                    new_bsdf_shader.inputs["Emission Color"].default_value = normal_map_node.inputs[
                                        'Color'].default_value
                                else:
                                    new_bsdf_shader.inputs["Emission"].default_value = normal_map_node.inputs[
                                        'Color'].default_value
                        else:
                            if version_info[0] > 3:
                                new_bsdf_shader.inputs["Emission Color"].default_value = (0.5, 0.5, 1, 1)
                            else:
                                new_bsdf_shader.inputs["Emission"].default_value = (0.5, 0.5, 1, 1)

                    if "Alpha" in node.inputs:
                        if node.inputs["Alpha"].is_linked:
                            color_origin_socket = node.inputs["Alpha"].links[0].from_socket
                            node_tree.links.new(color_origin_socket, new_bsdf_shader.inputs['Alpha'])
                        else:
                            new_bsdf_shader.inputs['Alpha'].default_value = node.inputs["Alpha"].default_value


def change_shader_emission(object, from_material_name: str = 'Base Color'):
    """
    change shader emission color from "from_material_name" and disable original input
    :param object: blender object to change
    :param from_material_name: name of the input slot on BSDF node to connect to emission
    """
    version_info = bpy.app.version
    if from_material_name != 'Base Color':
        remove_image_linkage(object, material_input_type='Base Color', remove_tex_image=False)
    if version_info[0] > 3:
        remove_image_linkage(object, material_input_type='Emission Color', remove_tex_image=False)
    else:
        remove_image_linkage(object, material_input_type='Emission', remove_tex_image=False)

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            links = node_tree.links
            if from_material_name != 'Base Color':
                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs['Base Color'].links) > 0:
                            l = node.inputs['Base Color'].links[0]
                            if l.from_socket.name == 'Color':
                                tex_image_node = l.from_node
                                if tex_image_node.type == 'TEX_IMAGE':
                                    links.remove(l)
                        node.inputs["Base Color"].default_value = (0, 0, 0, 1)

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if len(node.inputs[from_material_name].links) > 0:
                        l = node.inputs[from_material_name].links[0]
                        if l.from_socket.name == 'Color':
                            tex_image_node = l.from_node
                            if tex_image_node.type == 'TEX_IMAGE':
                                links.remove(l)
                                if version_info[0] > 3:
                                    links.new(tex_image_node.outputs["Color"], node.inputs["Emission Color"])
                                    node.inputs["Emission Strength"].default_value = 1
                                else:
                                    links.new(tex_image_node.outputs["Color"], node.inputs["Emission"])
                                    node.inputs["Emission Strength"].default_value = 1
                    else:
                        material_value = node.inputs[from_material_name].default_value
                        if type(material_value) is float:
                            material_value = (
                                material_value, material_value, material_value, 1)
                        node.inputs["Emission Strength"].default_value = 1
                        if version_info[0] > 3:
                            node.inputs["Emission Color"].default_value = material_value
                        else:
                            node.inputs["Emission"].default_value = material_value


def sphere_point_sample(n=300, object_center=np.zeros(3, dtype=np.float32), radius=1.0):
    """
    Sample points on a unit sphere.
    We use fibonacci spiral, quasi-uniform sample points distribution.
    See https://extremelearning.com.au/evenly-distributing-points-on-a-sphere/ for more information.
    :param n: number of points to sample
    :param object_center: the center of the sphere
    :param radius: the radius of the sphere
    :returns: a numpy array of n sampled points (shape is n*3)
    """
    pi = 3.14
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = arange(0, n)
    theta = 2 * pi * i / goldenRatio
    phi = arccos(1 - 2 * (i + 0.5) / n)
    x, y, z = radius * cos(theta) * sin(phi), radius * \
              sin(theta) * sin(phi), radius * cos(phi)
    return np.stack([x, y, z], axis=-1) + object_center[None, ...]


# sample points on unit torus
def torus_point_sample(n=100, object_center=np.zeros(3, dtype=np.float32), radius=1.0):
    """
    sample points on a unit torus, the center of the torus is the object center
    :param n: number of points to sample
    :param object_center: the center of the torus
    :param radius: the radius of the torus
    :returns: a numpy array of n sampled points (shape is n*3)
    """
    pi = 3.14
    i = arange(0, n)
    theta = 2 * pi * i / float(n + 1)

    phi = math.pi / 2 * np.ones(n, dtype=float)
    x, y, z = radius * cos(theta) * sin(phi), radius * \
              sin(theta) * sin(phi), radius * cos(phi)

    return np.stack([x, y, z], axis=-1) + object_center[None, ...]


def write_done(path: str):
    """
    write task.done file to render folder indicates that rendering process is finished
    :param path: path to render folder
    """
    file_name = "task.done"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            try:
                with open(file_fullpath, 'w') as fs:
                    fs.write("done")
            except (IOError, FileNotFoundError):
                logging.error(f"Cannot write done file {file_fullpath}")


# change transformation materix in opencv coordinates to blender coordinates
def opencv_to_blender(T):
    """
    change transformation materix in opencv coordinates to blender coordinates
    transform a point like: new_point = np.matmul(output_transform, old_point)
    :param T: transformation matrix in opencv coordinate system, 4 * 4 numpy array
    :returns: transformation matrix in blender coordinate system, 4 * 4 numpy array
    """
    origin = np.array(((1, 0, 0, 0),
                       (0, -1, 0, 0),
                       (0, 0, -1, 0),
                       (0, 0, 0, 1)))
    return np.matmul(T, origin)  # T * origin


def blender_to_opencv(T):
    """
    change transformation materix in blender coordinates to opencv coordinates
    transform a point like: new_point = np.matmul(output_transform, old_point)
    :param T: transformation matrix in blender coordinate system, 4 * 4 numpy array
    :returns: transformation matrix in opencv coordinate system, 4 * 4 numpy array
    """
    transform = np.array(((1, 0, 0, 0),
                          (0, -1, 0, 0),
                          (0, 0, -1, 0),
                          (0, 0, 0, 1)))
    return np.matmul(T, transform)  # T * transform


def look_at(obj_camera, point):
    """
    calculate lookat matrix of a camera
    :param obj_camera: blender camera object
    :param point: point to look at (mostly center of a sphere)
    """
    loc_camera = obj_camera.location
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def set_camera(bpy_cam, angle=3.14 / 3, W=600, H=500):
    """
    set camera matrix in blender
    :param bpy_cam: blender camera object
    :param angle: camera fov
    :param W: width of the output image
    :param H: height of the output image
    """
    bpy_cam.angle = angle
    bpy_scene = bpy.context.scene
    bpy_scene.render.resolution_x = W
    bpy_scene.render.resolution_y = H


def write_list(list_path: str, write_list: list):
    """
    write list to a file
    :param list_path: path to write file
    :param write_list: list to write
    """
    try:
        with open(list_path, 'w') as f:
            for index in range(len(write_list)):
                f.write(write_list[index] + "\n")
    except (IOError, FileNotFoundError):
        logging.error("Cannot write list file %s" % list_path)


def numpy_array_2_xyz(np_array_1, xyz_filepath: str):
    """
    convert numpy array containing point cloud to xyz file (supported by MeshLab)
    :param np_array_1: numpy array
    :param xyz_filepath: path to write xyz file
    """
    xyz_list = []
    for index in range(np_array_1.shape[0]):
        xyz_list.append(str(
            np_array_1[index][0]) + " " + str(np_array_1[index][1]) + " " + str(np_array_1[index][2]))

    write_list(xyz_filepath, xyz_list)


def join_list_of_mesh(mesh_list: list):
    """
    join a list of meshes into a single mesh
    :param mesh_list: list of meshes
    """
    if len(mesh_list) <= 0:
        raise ValueError("mesh_list must contain at least one element")
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


def mesh_shade_smooth(the_mesh):
    """
    set mesh to smooth shading in blender
    :param the_mesh: mesh to set smooth shading
    """
    version_info = bpy.app.version
    # blender changes shade smooth api after version 4.2.x
    if version_info[0] >= 4 and version_info[1] >= 1:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = the_mesh
        the_mesh.select_set(True)
        bpy.ops.object.shade_auto_smooth(angle=0.523599)
        bpy.ops.object.select_all(action='DESELECT')
    else:
        the_mesh.data.use_auto_smooth = True
        the_mesh.data.auto_smooth_angle = math.radians(30)
    for f in the_mesh.data.polygons:
        f.use_smooth = True


def compute_mesh_size(meshes):
    """
    compute total size of a list of meshes
    :param meshes: list of meshes
    :return obj_center: center of the mesh
    :return length: longest edge length of the mesh's AABB
    :return diagonal: diagonal length of the mesh
    :return min_: minimum point of the mesh
    :return max_: maximum point of the mesh
    """
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    verts = []
    for ind, mesh in enumerate(meshes):
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        current_vertices = [(v.co) for v in bm.verts]
        current_vertices = np.asarray(current_vertices, dtype=np.float32)
        bm.free()
        bpy.ops.object.editmode_toggle()

        mat = np.asarray(mesh.matrix_world)
        R, t = mat[:3, :3], mat[:3, 3:]  # Apply World Scale
        verts.append((R @ current_vertices.T + t).T)
    verts = np.concatenate(verts, axis=0)

    min_0 = verts[:, 0].min(axis=0)
    max_0 = verts[:, 0].max(axis=0)
    min_1 = verts[:, 1].min(axis=0)
    max_1 = verts[:, 1].max(axis=0)
    min_2 = verts[:, 2].min(axis=0)
    max_2 = verts[:, 2].max(axis=0)

    min_ = np.array([min_0, min_1, min_2])
    max_ = np.array([max_0, max_1, max_2])

    obj_center = (min_ + max_) / 2

    # use max len of xyz, instead of z
    length = max(max_ - min_)
    diagonal = np.linalg.norm(max_ - min_)

    return obj_center, length, diagonal, min_, max_, verts


def connect_color_to_base_color(object):
    """
    connect color attributes to base color
    :param object: blender object that contains color attributes
    """
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
                            links.remove(l)

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    color_attribute_node = nodes.new(type='ShaderNodeVertexColor')
                    links.new(color_attribute_node.outputs["Color"], node.inputs["Base Color"])


def create_material(object, material_name: str):
    """
    create a new material and assign it to object
    :param object: blender object to add material
    :param material_name: name of the newly created material
    :return: the newly created material
    """
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


def remove_default_value(object, material_input_type: str = "Emission Strength"):
    """
    remove default value of a slot in BSDF node
    :param object: blender object contains the material to remove default value
    :param material_input_type: the name of input slot to be removed
    """
    if object.material_slots:
        for slot in object.material_slots:
            set_node_default_value(slot.material, 0.0, material_input_type)


def set_node_default_value(material, toset_value: float, material_input_type: str = "Base Color"):
    """
    set default value of an input slot in BSDF node
    :param material: the material to remove default value
    :param toset_value: the value to be set
    :param material_input_type: the name of input slot to be removed
    """
    version_info = bpy.app.version
    node_tree = material.node_tree
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            if version_info[0] >= 4:
                if material_input_type == "Specular":
                    material_input_type = "Specular IOR Level"
            node.inputs[material_input_type].default_value = toset_value


def set_default_value(object, toset_value: float, material_input_type: str = "Metallic"):
    """
    set default value of all the materials using set_node_default_value(...)
    :param object: blender object to remove default value
    :param toset_value: the value to be set
    :param material_input_type: the name of input slot to be removed
    """
    if object.material_slots:
        for slot in object.material_slots:
            set_node_default_value(slot.material, toset_value, material_input_type)


def render_process(character_path: str,
                   dump_path: str,
                   transform_path: str,
                   pose_json_path: str,
                   hdr_file_path: str,
                   H: int,
                   W: int,
                   engine_type: str = "eevee",
                   render_device: str = "GPU",
                   aux_image_type: str = "png",
                   render_config_path: str = "",
                   only_render_png: bool = False,
                   render_prefix: str = "",
                   use_solidify: bool = False,
                   use_smooth: bool = False,
                   parse_exr: bool = False,
                   use_point_light: bool = False,
                   render_daz: bool = False,
                   render_material: bool = False,
                   material_type: str = 'PBR',
                   hdr_strength: float = 1.0,
                   render_equilibrium: bool = False,
                   use_outside_transform: bool = False,
                   use_unit_transform: bool = False,
                   no_camera_export: bool = False,
                   use_better_fbx: bool = False,
                   export_scaled_obj: bool = False,
                   emission_render: bool = False,
                   rotate_object: bool = False,
                   colored_background: bool = False,
                   use_color_attribute: bool = False,
                   debug_blend_save: bool = False):
    """
    Render a mesh.
    :param character_path: path to the input mesh file
    :param dump_path: path to the output directory
    :param transform_path: path to the transformation.txt file,
                           which stores the transformation that scales object to unit sphere.
                           when use_outside_transform is true, transform_path is input;
                           otherwise it's output.
    :param pose_json_path: path to the render camera pose json file
    :param H: height of the output image
    :param W: width of the output image
    :param engine_type: blender engine to be used,
                        before blender 4.2 could be cycles/eevee,
                        after blender 4.2 could be cycles/eevee_next
    :param render_device: render device to be used, GPU or CPU
    :param aux_image_type: aux data (normal/depth/xyz) type,
                           png: uint8 image/png_16bit: uint16 image/tif: float16 image
    :param render_config_path: path to the render configuration file
    :param only_render_png: only render color image as png format
    :param render_prefix: prefix of the output image
    :param use_solidify: use solidify modifier in blender to make mesh thicker
    :param use_smooth: add shade_smooth to objects in blender
    :param parse_exr: parse exr file and extract normal/xyz/depth.....
    :param use_point_light: sample a group of points on sphere and add point light to these points as light source
    :param render_daz: fix axis error when the mesh is actually z-up
    :param render_material: render PBR or bump
    :param material_type: the type of material to be rendered, choose between PBR/bump
    :param render_equilibrium: render light source is an exactly equally distributed hdri map,
                               and remove specular and material
    :param use_outside_transform: use outside transformation.txt file
    :param use_unit_transform: ignore all internal or external object transformation
    :param no_camera_export: do not output cam_parameters.json file
    :param use_better_fbx: use very slow but more robust better_fbx import plugin to import fbx file
    :param export_scaled_obj: export mesh after scaling as obj file
    :param emission_render: render base color using emission, this can be used to produce raw texture
    :param rotate_object: rotate the object during rendering
    :param colored_background: make the background of rendered image has color
    :param use_color_attribute: connect color attributes to base_color input slot of BSDF node
    :param debug_blend_save: save blend file for reproducing render scene
    """

    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    with open(pose_json_path, encoding='utf-8') as f:
        camera_parameters_data = json.load(f)
        cam_poses = camera_parameters_data["poses"]
        if "object_pose" not in camera_parameters_data.keys():
            rotate_object = False
        if rotate_object:
            object_poses = camera_parameters_data["object_poses"]

    render_config_data = None
    if len(render_config_path) > 1:
        if os.path.exists(render_config_path):
            with open(render_config_path, encoding='utf-8') as f:
                render_config_data = json.load(f)

    # set some trivial default value
    if render_config_data is None:
        background_certain_color = [1.0, 1.0, 1.0, 1.0]
        n_lights = 25
        point_light_energy = 5.0
        solidify_thickness = 0.005
        standard_height = 1.92
    else:
        background_certain_color = render_config_data["background_certain_color"]
        n_lights = render_config_data["render_lights_number"]
        point_light_energy = render_config_data["light_energy"]
        solidify_thickness = render_config_data["solidify_thickness"]
        if "standard_height" not in render_config_data.keys():
            standard_height = 1.92
        else:
            standard_height = render_config_data["standard_height"]

    exr_dump_path = os.path.join(dump_path, "exr")
    obj_dum_path = os.path.join(dump_path, "obj")
    color_dump_path = os.path.join(dump_path, "color")

    if not os.path.exists(exr_dump_path):
        os.mkdir(exr_dump_path)

    if export_scaled_obj:
        if not os.path.exists(obj_dum_path):
            os.mkdir(obj_dum_path)

    if not os.path.exists(color_dump_path):
        os.mkdir(color_dump_path)

    # remove all default objects in blender
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # avoid effects of some default light in blender world, and set output color space
    if emission_render:
        bpy.context.scene.view_settings.view_transform = 'Standard'
        clear_blender_world()
    if render_material:
        bpy.context.scene.view_settings.view_transform = 'Raw'
        clear_blender_world()

    character_type = os.path.splitext(character_path)[-1].lower()
    if ".fbx" == character_type:
        if not use_better_fbx:
            try:
                bpy.ops.import_scene.fbx(filepath=character_path, use_anim=True)
            except:
                bpy.ops.preferences.addon_enable(module="better_fbx")
                bpy.ops.better_import.fbx(filepath=character_path, use_optimize_for_blender=False,
                                          use_auto_bone_orientation=True,
                                          use_reset_mesh_origin=True, use_reset_mesh_rotation=True,
                                          use_detect_deform_bone=True, use_auto_smooth=True,
                                          use_animation=True)
        else:
            try:
                bpy.ops.preferences.addon_enable(module="better_fbx")
                bpy.ops.better_import.fbx(filepath=character_path, use_optimize_for_blender=False,
                                          use_auto_bone_orientation=True,
                                          use_reset_mesh_origin=True, use_reset_mesh_rotation=True,
                                          use_detect_deform_bone=True, use_auto_smooth=True,
                                          use_animation=True)
            except:
                bpy.ops.import_scene.fbx(filepath=character_path, use_anim=True)

    elif ".obj" == character_type:
        version_info = bpy.app.version
        if version_info[0] > 2:
            if render_daz:
                bpy.ops.wm.obj_import(filepath=mesh_path,
                                      forward_axis='Y', up_axis='Z')
            else:
                bpy.ops.wm.obj_import(filepath=mesh_path)
        else:
            if render_daz:
                bpy.ops.import_scene.obj(
                    filepath=character_path, axis_forward='Y', axis_up='Z')
            else:
                bpy.ops.import_scene.obj(filepath=character_path)
    elif ".vrm" == character_type:
        addon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'addons/VRM_Addon_for_Blender-release.zip')
        bpy.ops.preferences.addon_install(overwrite=True, target='DEFAULT', filepath=addon_path,
                                          filter_folder=True, filter_python=False, filter_glob="*.py;*.zip")
        bpy.ops.preferences.addon_enable(module="VRM-Addon-for-Blender-README")
        bpy.ops.wm.vrm_license_warning(
            filepath=character_path,
            license_confirmations=[{"name": "LicenseConfirmation0",
                                    "message": "This VRM is licensed by VRoid Hub License \"Alterations: No\".",
                                    "url": "https://hub.vroid.com/license?allowed_to_use_user=everyone&characterization_allowed_user=author&corporate_commercial_use=disallow&credit=necessary&modification=disallow&personal_commercial_use=disallow&redistribution=disallow&sexual_expression=disallow&version=1&violent_expression=disallow",
                                    "json_key": "otherPermissionUrl"},
                                   {"name": "LicenseConfirmation1",
                                    "message": "This VRM is licensed by VRoid Hub License \"Alterations: No\".",
                                    "url": "https://hub.vroid.com/license?allowed_to_use_user=everyone&characterization_allowed_user=author&corporate_commercial_use=disallow&credit=necessary&modification=disallow&personal_commercial_use=disallow&redistribution=disallow&sexual_expression=disallow&version=1&violent_expression=disallow",
                                    "json_key": "otherLicenseUrl"}],
            import_anyway=True, extract_textures_into_folder=False, make_new_texture_folder=True)
    elif ".glb" == character_type:
        bpy.ops.import_scene.gltf(filepath=character_path)
    elif ".blend" == character_type:
        bpy.ops.wm.open_mainfile(filepath=character_path)
    else:
        logging.error(f"mesh data format {character_type} not supported")
        raise ValueError("mesh data format {character_type} not supported")

    # Remove weapon mesh
    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    size_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    try:
        # switch character to rest mode, i.e. A-pose in most case
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.loc_clear()
        bpy.ops.pose.rot_clear()
        bpy.ops.pose.scale_clear()
        bpy.ops.object.posemode_toggle()
    except:
        # just pass this if pose toggle failed
        logging.warning('posemode_toggle failed')

    # force to connect color attributes onto BSDF's base color input
    # this is necessary because the color attribute is not connected by default in obj and fbx formats
    if use_color_attribute:
        for mesh in meshes:
            create_material(object=mesh, material_name="new_material")
            if bpy.app.version[0] > 3:
                remove_default_value(mesh, material_input_type="Specular IOR Level")
            else:
                remove_default_value(mesh, material_input_type="Specular")
            connect_color_to_base_color(mesh)

    obj_center, length, diagonal, max_point, min_point, mesh_verts = compute_mesh_size(meshes)
    logging.info("Object initial center is %s; max point is %s; min point is %s......." % (
        str(obj_center), str(max_point), str(min_point)))

    if not use_unit_transform:
        if use_outside_transform and len(transform_path) > 1 and os.path.exists(transform_path):
            for mesh in meshes:
                T = np.loadtxt(transform_path)
                logging.info(f"outside transformation is {str(T)}")
        # we first calculate a tight bounding sphere of the mesh
        # and scale the diameter of this sphere to standard_height
        # we don't choose AABB because
        # (1) is not as tight as a bounding sphere
        # (2) if you rotate an object scaled within a AABB, one part of the object may be out of the unit volume
        else:
            # calculate convex hull of the mesh
            # some meshes have many vertices and will be very slow when calculating bounding sphere
            original_mesh = trimesh.base.Trimesh(vertices=mesh_verts)
            hull_vertices = original_mesh.convex_hull.vertices
            # library miniball may fail to calculate bounding sphere
            # but we cannot stably reproduce the failure, as it disappears if we re-run the calculate code
            # so calculate three times is a temporary workaround
            try:
                bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
            except:
                # it's necessary to block the code for 0.1 seconds
                time.sleep(0.1)
                logging.warning("Miniball failed. Retry once...............")
                try:
                    bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
                except:
                    time.sleep(0.1)
                    logging.warning("Miniball failed. Retry second time...............")
                    try:
                        bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
                    except:
                        logging.error("Mniball failed multiple times, aborting....")
                        raise ArithmeticError("Mniball failed multiple times, data may contain errors....")

            obj_center = bounding_sphere_C
            length = 2 * math.sqrt(bounding_sphere_r2)
            trn = -1 * obj_center[..., np.newaxis]
            for mesh in meshes:
                transform_path = os.path.join(dump_path, "transformation.txt")
                scale = standard_height / length
                T = np.eye(4)
                T[:3, 3:] = scale * trn
                T[:3, :3] = scale * T[:3, :3]
                logging.info(f"inside transformation is {str(T)}")
                np.savetxt(transform_path, T)
                mesh.matrix_world = Matrix(T) @ mesh.matrix_world

    # add solidify modifier to the mesh
    if use_solidify:
        for m in meshes:
            m.modifiers.new("solidify", "SOLIDIFY")
            m.modifiers["solidify"].thickness = solidify_thickness
            meshd = m.data
            for f in meshd.polygons:
                f.use_smooth = False

    if use_smooth:
        for m in meshes:
            mesh_shade_smooth(m)

    # removes "bright" parts in the material
    if render_equilibrium:
        emission_render = False
        render_material = False
        for m in meshes:
            remove_image_linkage(m, material_input_type="Metallic")
            remove_image_linkage(m, material_input_type="Specular")
            set_default_value(m, 0.0, material_input_type="Metallic")
            set_default_value(m, 0.0, material_input_type="Specular")

    # add color to background of rendered image; default is [0,0,0,0]
    if colored_background:
        # override these options
        emission_render = False
        only_render_png = True

        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_settings.view_transform = 'Standard'
        composite_node_tree = bpy.context.scene.node_tree
        composite_alpha_node = composite_node_tree.nodes.new("CompositorNodeAlphaOver")
        composite_alpha_node.premul = 1.0
        composite_alpha_node.inputs[1].default_value = (background_certain_color[0],
                                                        background_certain_color[1],
                                                        background_certain_color[2],
                                                        background_certain_color[3])
        for node in composite_node_tree.nodes:
            if node.type == "R_LAYERS":
                if len(node.outputs["Image"].links) > 0:
                    l = node.outputs["Image"].links[0]
                    final_output_node = l.to_node
                    if l is not None:
                        composite_node_tree.links.remove(l)
                    composite_node_tree.links.new(node.outputs["Image"], composite_alpha_node.inputs[2])
                    composite_node_tree.links.new(composite_alpha_node.outputs["Image"],
                                                  final_output_node.inputs["Image"])

    scaled_mesh_filename = os.path.join(obj_dum_path, "scaled.obj")
    # export scaled mesh to obj folder; only for debugging purpose
    if export_scaled_obj:
        bpy.ops.wm.obj_export(filepath=scaled_mesh_filename, path_mode='COPY')

    camera_points_list = []
    render_camera_data = {}
    render_object_data = {}
    all_camera_names = list(cam_poses.keys())

    # same with thumbnail names in generate_thumbnail.py
    thumbnail_names = [100, 22, 121, 41, 268, 85, 291, 68, 91]

    for index in range(len(all_camera_names)):
        camera_name = all_camera_names[index]

        camera_model_type = "perspective"
        if "model" in cam_poses[camera_name].keys():
            camera_model_type = cam_poses[camera_name]["model"]

        pose = np.array(cam_poses[camera_name]["pose"])
        R_list = pose[0:3, 0:3].tolist()
        R = Matrix(R_list).to_euler('XYZ')
        t = Vector(pose[0:3, 3])
        camera_points_list.append(t)

        camera_data = bpy.data.cameras.new(name=camera_name)
        camera_object = bpy.data.objects.new(camera_name, camera_data)
        bpy.context.scene.collection.objects.link(camera_object)
        bpy.context.scene.render.resolution_x = W
        bpy.context.scene.render.resolution_y = H

        camera_data.display_size = standard_height * 0.1
        camera_data.clip_start = 0.01
        camera_data.clip_end = 100
        camera_object.location = t
        camera_object.rotation_euler = R

        if camera_model_type == "perspective":
            camera_data.type = 'PERSP'
            K = np.array(cam_poses[camera_name]["k"])
            camera_angle = 2.0 * np.arctan2(K[0][2], K[0][0])
            pose_opencv = blender_to_opencv(pose)
            render_camera_data[camera_name] = {
                "k": np.asarray(K).tolist(),
                "pose": pose_opencv.tolist(),
                "scale": None,
                "model": "perspective"
            }
            camera_data.angle = camera_angle
        elif camera_model_type == "orthographic":
            camera_data.type = 'ORTHO'
            ortho_scale = float(cam_poses[camera_name]["scale"])
            pose_opencv = blender_to_opencv(pose)
            render_camera_data[camera_name] = {
                "k": None,
                "pose": pose_opencv.tolist(),
                "scale": ortho_scale,
                "model": "orthographic"
            }
            camera_data.ortho_scale = ortho_scale
        # update camera params manually; ensure that camera is updated
        bpy.context.view_layer.update()

    camera_points_array = np.array(camera_points_list)
    store_pose_json_path = os.path.join(dump_path, "cam_parameters.json")
    if not no_camera_export:
        json_object = json.dumps(render_camera_data, indent=4)
        with open(store_pose_json_path, "w") as outfile:
            outfile.write(json_object)
        numpy_array_2_xyz(camera_points_array, os.path.join(dump_path, "camera.xyz"))

    # no light is used in emission_render or material_render
    if not emission_render and not render_material:
        if use_point_light:
            lights_center = sphere_point_sample(n_lights)
            lights_center = lights_center * standard_height  # scale and translate
            for i in range(n_lights):
                bpy.ops.object.light_add(type='POINT',
                                         radius=np.random.normal(standard_height, standard_height * 0.1),
                                         align='WORLD',
                                         location=Vector(lights_center[i]),
                                         scale=(1, 1, 1))
                bpy.context.object.data.energy = point_light_energy

    if engine_type == "cycles":
        bpy.context.scene.render.engine = 'CYCLES'
    elif engine_type == "eevee":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    elif engine_type == "eevee_next":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    else:
        logging.error("not supported engine_type")
        raise ValueError("not supported engine_type")

    if emission_render:
        for mesh in meshes:
            change_shader_emission(mesh, from_material_name='Base Color')

    if render_material:
        for mesh in meshes:
            fix_bump_color_space(mesh)
            if material_type == "PBR":
                fix_material_space(mesh, input_type="Metallic")
                fix_material_space(mesh, input_type="Roughness")
                pbr_render_shader(mesh)
            elif material_type == 'bump':
                bump_render_shader(mesh)
            else:
                logging.error(f"Unsupported material type {material_type}")
                raise ValueError(f"Unsupported material type {material_type}")
    else:
        for mesh in meshes:
            fix_bump_color_space(mesh)
            fix_material_space(mesh, "Metallic")
            fix_material_space(mesh, "Specular")
            fix_material_space(mesh, "Roughness")

    if engine_type == "cycles":
        if render_material:
            # disable denoising for emission render to reduce render time
            # sample number set to 1 is from
            # https://github.com/3DTopia/MaterialAnything/blob/main/rendering_scripts/blender_script_material.py
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.samples = 1
        elif emission_render:
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.samples = 1024
        else:
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.samples = 4096

        if render_device == "GPU":
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                if 'Intel' in d["name"] or 'AMD' in d["name"]:
                    d["use"] = 0
                else:
                    d["use"] = 1
                logging.info(d["name"], ",", d["id"], ",", d["type"], ",", d["use"])

            if only_render_png:
                bpy.context.scene.render.image_settings.color_depth = '8'
                bpy.context.scene.render.image_settings.file_format = 'PNG'
            else:
                bpy.context.scene.render.image_settings.color_depth = '16'
                bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        else:
            bpy.context.scene.cycles.device = 'CPU'
            if only_render_png:
                bpy.context.scene.render.image_settings.color_depth = '8'
                bpy.context.scene.render.image_settings.file_format = 'PNG'
            else:
                bpy.context.scene.render.image_settings.color_depth = '16'
                bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    else:
        bpy.context.scene.render.use_high_quality_normals = True
        if only_render_png:
            bpy.context.scene.render.image_settings.color_depth = '8'
            bpy.context.scene.render.image_settings.file_format = 'PNG'
        else:
            bpy.context.scene.render.image_settings.color_depth = '16'
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.use_pass_normal = True
    if engine_type != "eevee":
        bpy.context.view_layer.use_pass_position = True

    # save reproducible blend file
    if debug_blend_save:
        debug_blend_file = os.path.join(dump_path, "debug.blend")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=debug_blend_file,
                                    compress=False,
                                    check_existing=False)

    change_hdr_map_path(hdr_map_path=hdr_file_path, strength=hdr_strength)

    # change hdr map will rebuild light volume, which consumes a lot of memory
    # record previous hdr map path and if it's the same with current, skip replacement function
    previous_hdr_path = None
    for index in range(len(all_camera_names)):
        camera_name = all_camera_names[index]
        bpy.context.scene.camera = bpy.data.objects[camera_name]
        # rotate object if necessary
        if rotate_object:
            object_pose = np.array(object_poses[camera_name]["object_pose"])
            object_pose_opencv = blender_to_opencv(object_pose)
            object_R_list = object_pose_opencv[0:3, 0:3].tolist()
            object_R = Matrix(object_R_list).to_euler('XYZ')
            render_object_data[camera_name] = {
                "pose": object_pose_opencv.tolist()
            }
            for m in meshes:
                m.rotation_euler[0] = m.rotation_euler[0] + object_R[0]
                m.rotation_euler[1] = m.rotation_euler[1] + object_R[1]
                m.rotation_euler[2] = m.rotation_euler[2] + object_R[2]
                bpy.context.view_layer.update()
                time.sleep(0.1)

        if only_render_png:
            render_result_filepath = os.path.join(color_dump_path, str(render_prefix) + str(camera_name) + ".png")
        else:
            render_result_filepath = os.path.join(exr_dump_path, str(render_prefix) + str(camera_name) + ".exr")

        bpy.context.scene.render.filepath = render_result_filepath
        bpy.ops.render.render(write_still=True)

        # we use separate exr parser script because install OpenEXR in blender python is not easy
        if parse_exr:
            if not only_render_png:
                exr_parser_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exr_parser.py")
                cmd = "python {} --exr_file \"{}\" --camera_info_path \"{}\" --remove_exr ".format(
                    exr_parser_script_path, render_result_filepath, pose_json_path)
                cmd = cmd + " --format {} ".format(aux_image_type)
                if not run_cmd(cmd):
                    logging.error(f"Cannot parse exr file at {render_result_filepath}")
                    raise ValueError(f"Wrong exr file at {render_result_filepath}")

    if rotate_object:
        store_object_pose_json_path = os.path.join(dump_path, "object_parameters.json")
        object_json_object = json.dumps(render_object_data, indent=4)
        with open(store_object_pose_json_path, "w") as outfile:
            outfile.write(object_json_object)

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


if __name__ == '__main__':
    time_point1 = time.time()
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Render one mesh script.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--output_folder', type=str,
                        default="", help='render result output folder')
    parser.add_argument('--transform_path', type=str, default="",
                        help='transform txt file path...')
    parser.add_argument('--pose_json_path', type=str,
                        help='pose json file path...')
    parser.add_argument('--camera_config_path', type=str, default="",
                        help='config for each camera in rendering...')
    parser.add_argument('--hdr_file_path', type=str, default="",
                        help='hdr file used as background light and color')
    parser.add_argument('--hdr_strength', type=float, default=0.5,
                        help='hdr strength used in blender\'s world system')
    parser.add_argument('--parse_exr', action='store_true',
                        help='parser exr in render process, by default do not parse')
    parser.add_argument('--render_daz', action='store_true',
                        help='render daz obj file which uses different coordinate system')
    parser.add_argument('--only_render_png', action='store_true',
                        help='only render color image')
    parser.add_argument('--solidify', action='store_true',
                        help='use solidify to eliminate some artifacts')
    parser.add_argument('--smooth', action='store_true',
                        help='smooth mesh rendering images')
    parser.add_argument('--file_prefix', type=str, default="",
                        help='add a prefix to all render output files')
    parser.add_argument('--engine', type=str, default="eevee",
                        help='render engine in blender, can be eevee/eevee_next/cycles')
    parser.add_argument('--render_device', type=str, default="GPU",
                        help='render device when using cycles renderer')
    parser.add_argument('--render_height', type=int,
                        default=512, help='height of render output image')
    parser.add_argument('--render_width', type=int,
                        default=512, help='width of render output image')
    parser.add_argument('--aux_image_type', type=str, default="png",
                        help='aux data (normal/depth/xyz) type, can be png/tif/png_16bit')
    parser.add_argument('--light_source', type=str, default="hdr",
                        help='light source used in rendering, '
                             '"hdr" will use a pre-defined hdr map; '
                             '"light" will use blender internal light source '
                             'and light number is determined by render_lights_number')
    parser.add_argument('--render_material', action='store_true',
                        help='render other material entries such as roughness and material')
    parser.add_argument('--material_type', type=str, default="Roughness",
                        help='name of material entry')
    parser.add_argument('--render_equilibrium', action='store_true',
                        help='render light source is an exactly equally distributed hdri map, '
                             'and remove specular and material')
    parser.add_argument('--use_outside_transform', action='store_true',
                        help='use outside transform txt file...')
    parser.add_argument('--use_unit_transform', action='store_true',
                        help='use unit transform inside, ignore all internal and external transformation.txt...')
    parser.add_argument('--no_camera_export', action='store_true',
                        help='no export camera parameters json and xyz file...')
    parser.add_argument('--use_better_fbx', action='store_true',
                        help='use very slow but more robust better_fbx import plugin...')
    parser.add_argument('--export_scaled_obj', action='store_true',
                        help='export scaled mesh results in obj format...')
    parser.add_argument('--only_emission', action='store_true',
                        help='connect texture image to emission node to produce raw texture...')
    parser.add_argument('--rotate_object', action='store_true',
                        help='rotate object during rendering')
    parser.add_argument('--colored_background', action='store_true',
                        help='force to use colored background in rendering output...')
    parser.add_argument('--use_color_attribute', action='store_true',
                        help='force to use color attributes as input of bsdf node...')
    parser.add_argument('--debug_blend_save', action='store_true',
                        help='save blend file in render process for debug...')

    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    pose_json_path = args.pose_json_path
    hdr_file_path = args.hdr_file_path
    hdr_strength = args.hdr_strength
    camera_config_path = args.camera_config_path
    output_folder = args.output_folder
    transform_path = args.transform_path
    parse_exr = args.parse_exr
    render_daz = args.render_daz
    render_material = args.render_material
    material_type = args.material_type
    render_equilibrium = args.render_equilibrium
    only_render_png = args.only_render_png
    render_height = args.render_height
    render_width = args.render_width
    aux_image_type = args.aux_image_type
    file_prefix = args.file_prefix
    light_source = args.light_source
    engine_type = args.engine
    render_device = args.render_device
    solidify = args.solidify
    smooth = args.smooth
    use_outside_transform = args.use_outside_transform
    use_unit_transform = args.use_unit_transform
    no_camera_export = args.no_camera_export
    use_better_fbx = args.use_better_fbx
    export_scaled_obj = args.export_scaled_obj
    rotate_object = args.rotate_object
    only_emission = args.only_emission
    colored_background = args.colored_background
    use_color_attribute = args.use_color_attribute
    debug_blend_save = args.debug_blend_save

    if light_source == 'point':
        use_point_light = True
    else:
        use_point_light = False

    render_process(mesh_path,
                   output_folder,
                   transform_path,
                   pose_json_path,
                   hdr_file_path,
                   render_height,
                   render_width,
                   engine_type=engine_type,
                   render_device=render_device,
                   aux_image_type=aux_image_type,
                   render_config_path=camera_config_path,
                   only_render_png=only_render_png,
                   render_prefix=file_prefix,
                   parse_exr=parse_exr,
                   render_daz=render_daz,
                   render_material=render_material,
                   material_type=material_type,
                   render_equilibrium=render_equilibrium,
                   use_point_light=use_point_light,
                   use_solidify=solidify,
                   use_smooth=smooth,
                   use_outside_transform=use_outside_transform,
                   use_unit_transform=use_unit_transform,
                   no_camera_export=no_camera_export,
                   use_better_fbx=use_better_fbx,
                   export_scaled_obj=export_scaled_obj,
                   emission_render=only_emission,
                   rotate_object=rotate_object,
                   colored_background=colored_background,
                   use_color_attribute=use_color_attribute,
                   debug_blend_save=debug_blend_save,
                   hdr_strength=hdr_strength)

    time_point2 = time.time()
    logging.info("Rendering mesh %s uses %f" % (mesh_path, (time_point2 - time_point1)))
    write_done(output_folder)
