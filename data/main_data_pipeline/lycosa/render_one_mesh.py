import argparse
import gc
import json
import math
import os
import sys
import time

import bmesh
import bpy
import miniball
import numpy as np
import trimesh
from mathutils import Matrix, Vector
from numpy import arange, sin, cos, arccos

engine_type = "eevee"
device = "GPU"
hero = "hok"

render_height = 512
render_width = 512
render_camera_number = 300
render_lights_number = 20
only_render_png = False
file_prefix = ""


def fix_bump_color_space(object):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name

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


def change_shader_emission(object):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            # emission_node = nodes.new("ShaderNodeEmission")
            # emission_node.inputs['Strength'].default_value = 0.2
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(node.inputs)
                    if len(node.inputs["Base Color"].links) > 0:
                        l = node.inputs["Base Color"].links[0]
                        if l.from_socket.name == 'Color':
                            tex_image_node = l.from_node
                            if tex_image_node.type == 'TEX_IMAGE':
                                links.remove(l)
                                time.sleep(0.1)
                                if version_info[0] > 3:
                                    links.new(
                                        tex_image_node.outputs["Color"], node.inputs["Emission Color"])
                                    node.inputs["Base Color"].default_value = (
                                        0, 0, 0, 1)
                                    node.inputs["Emission Strength"].default_value = 1
                                else:
                                    links.new(
                                        tex_image_node.outputs["Color"], node.inputs["Emission"])
                                    node.inputs["Base Color"].default_value = (
                                        0, 0, 0, 1)
                    else:
                        if version_info[0] > 3:
                            node.inputs["Emission Color"].default_value = node.inputs['Base Color'].default_value
                        else:
                            node.inputs["Emission"].default_value = node.inputs['Base Color'].default_value


def sphere_point_sample(n=300, object_center=np.zeros(3, dtype=np.float32), radius=1.0):
    # use fibonacci spiral
    pi = 3.14
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = arange(0, n)
    theta = 2 * pi * i / goldenRatio
    phi = arccos(1 - 2 * (i + 0.5) / n)
    x, y, z = radius * cos(theta) * sin(phi), radius * \
              sin(theta) * sin(phi), radius * cos(phi)
    return np.stack([x, y, z], axis=-1) + object_center[None, ...]


def torus_point_sample(n=100, object_center=np.zeros(3, dtype=np.float32), radius=1.0):
    pi = 3.14
    i = arange(0, n)
    theta = 2 * pi * i / float(n + 1)

    phi = math.pi / 2 * np.ones(n, dtype=float)
    x, y, z = radius * cos(theta) * sin(phi), radius * \
              sin(theta) * sin(phi), radius * cos(phi)

    return np.stack([x, y, z], axis=-1) + object_center[None, ...]


def write_done(path: str):
    file_name = "task.done"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("done")


def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0),
                       (0, -1, 0, 0),
                       (0, 0, -1, 0),
                       (0, 0, 0, 1)))
    return np.matmul(T, origin)  # T * origin


def blender_to_opencv(T):
    transform = np.array(((1, 0, 0, 0),
                          (0, -1, 0, 0),
                          (0, 0, -1, 0),
                          (0, 0, 0, 1)))
    return np.matmul(T, transform)  # T * transform


def look_at(obj_camera, point):
    loc_camera = obj_camera.location
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def set_camera(bpy_cam, angle=3.14 / 3, W=600, H=500):
    bpy_cam.angle = angle
    bpy_scene = bpy.context.scene
    bpy_scene.render.resolution_x = W
    bpy_scene.render.resolution_y = H


def write_list(path: str, write_list: list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def numpy_array_2_xyz(np_array_1, xyz_filepath: str):
    xyz_list = []
    for index in range(np_array_1.shape[0]):
        xyz_list.append(str(
            np_array_1[index][0]) + " " + str(np_array_1[index][1]) + " " + str(np_array_1[index][2]))

    write_list(xyz_filepath, xyz_list)


def set_intrinsic(K, bpy_camera, H, W):
    '''
    Set camera intrinsic parameters given the camera calibration. This function is written refering to https://blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model (an inverse version)
    :param K camera intrinsic parameters
    :param bpy_camera: blender camera object
    :param bpy_scene: blender scene object
    '''
    fx, fy, cx, cy = K
    resolution_x_in_px = W
    resolution_y_in_px = H
    bpy_scene = bpy.context.scene
    bpy_scene.render.resolution_x = resolution_x_in_px
    bpy_scene.render.resolution_y = resolution_y_in_px
    scale = resolution_x_in_px / (2 * cx)
    bpy_scene.render.resolution_percentage = scale * 100
    bpy_scene.render.pixel_aspect_x = 1.0
    bpy_scene.render.pixel_aspect_y = 1.0
    # both in mm
    bpy_camera.data.sensor_width = fy * cx / (fx * cy)
    bpy_camera.data.sensor_height = 1  # does not matter
    s_u = resolution_x_in_px / bpy_camera.data.sensor_width
    s_v = resolution_y_in_px / bpy_camera.data.sensor_height
    # we will use the default blender camera focal length
    bpy_camera.data.lens = fx / s_u


def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
        # s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    # alpha_v = f_in_mm * s_v
    alpha_v = alpha_u
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels
    K = Matrix(
        ((alpha_u, skew, u_0),
         (0, alpha_v, v_0),
         (0, 0, 1)))
    return K


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


def mesh_shade_smooth(object):
    object.data.use_auto_smooth = True
    object.data.auto_smooth_angle = math.radians(30)
    for f in object.data.polygons:
        f.use_smooth = True


def add_area_light(location, rotation, energy: float = 50.0, size: float = 5.0, light_name: str = "main"):
    light_data2 = bpy.data.lights.new(name=light_name + "_data", type='AREA')
    light_data2.energy = energy
    light_data2.size = size
    light_object2 = bpy.data.objects.new(
        name=light_name + "_light", object_data=light_data2)
    light_object2.location = location
    light_object2.rotation_euler = rotation
    bpy.context.collection.objects.link(light_object2)


def add_sun_light(location, energy: float = 2.0, light_name: str = "sun"):
    light_data2 = bpy.data.lights.new(name=light_name + "_data", type='SUN')
    light_data2.energy = energy
    light_object2 = bpy.data.objects.new(
        name=light_name + "_light", object_data=light_data2)
    light_object2.location = location
    bpy.context.collection.objects.link(light_object2)


def read_mesh_to_ndarray(mesh, mode="Edit"):
    ''' read the vert coordinate of a deformed mesh
    :param mesh: mesh object
    :return: numpy array of the mesh
    '''
    assert mode in ["edit", "object"]

    if mode == "object":
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(mesh, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        # faces = [[v.index for v in face.verts] for face in bm.faces]
        # faces = np.asarray(faces, dtype=np.int32)
        bm.free()
    elif mode == "edit":
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        # faces = [[v.index for v in face.verts] for face in bm.faces]
        # faces = np.asarray(faces, dtype=np.int32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co, None


def compute_mesh_size(meshes):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    verts = []
    for ind, mesh in enumerate(meshes):
        vert, _ = read_mesh_to_ndarray(mesh, mode="edit")
        mat = np.asarray(mesh.matrix_world)
        R, t = mat[:3, :3], mat[:3, 3:]  # Apply World Scale
        verts.append((R @ vert.T + t).T)
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


def set_node_default_value(material, toset_value, material_input_type: str = "Base Color"):
    node_tree = material.node_tree
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs[material_input_type].default_value = toset_value


def set_default_value(object, toset_value: float, material_input_type: str = "Metallic"):
    if object.material_slots:
        for slot in object.material_slots:
            set_node_default_value(
                slot.material, toset_value, material_input_type)


def oneExample_process(character_path: str, dump_path: str,
                       transform_path: str, pose_json_path: str,
                       H: int, W: int,
                       weapon_str_list: list,
                       n_lights: int = 25,
                       only_render_png: bool = False,
                       render_prefix: str = "",
                       use_solidify: bool = True,
                       use_smooth: bool = False,
                       parse_exr: bool = True,
                       render_daz: bool = False,
                       use_light: str = 'hdr',
                       hdr_map_path: str = "",
                       light_energy: float = 20,
                       solidify_thickness: float = 0.005,
                       render_thumbnail: bool = False,
                       use_outside_transform: bool = False,
                       use_unit_transform: bool = False,
                       no_camera_export: bool = False,
                       use_better_fbx: bool = False,
                       export_scaled_obj: bool = False,
                       emission_render: bool = False,
                       colored_background: bool = False,
                       debug_blend_save: bool = False,
                       background_certain_color: list = [1.0, 1.0, 1.0, 1.0]):
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    with open(pose_json_path, encoding='utf-8') as f:
        camera_parameters_data = json.load(f)
        cam_poses = camera_parameters_data["poses"]

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

    # remove all default objects in a collection
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    character_type = os.path.splitext(character_path)[-1].lower()
    if ".fbx" == character_type:
        if not use_better_fbx:
            try:
                bpy.ops.import_scene.fbx(
                    filepath=character_path, use_anim=True)
            except:
                addon_path = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), '../../addons/better_fbx.zip')
                print('addon_path ', addon_path)
                bpy.ops.preferences.addon_install(overwrite=True, target='DEFAULT', filepath=addon_path,
                                                  filter_folder=True, filter_python=False, filter_glob="*.py;*.zip")
                bpy.ops.preferences.addon_enable(module="better_fbx")
                bpy.ops.better_import.fbx(filepath=character_path, use_optimize_for_blender=False,
                                          use_auto_bone_orientation=True,
                                          use_reset_mesh_origin=True, use_reset_mesh_rotation=True,
                                          use_detect_deform_bone=True, use_auto_smooth=True,
                                          use_animation=True)
        else:
            try:
                addon_path = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), '../../addons/better_fbx.zip')
                print('addon_path ', addon_path)
                bpy.ops.preferences.addon_install(overwrite=True, target='DEFAULT', filepath=addon_path,
                                                  filter_folder=True, filter_python=False, filter_glob="*.py;*.zip")
                bpy.ops.preferences.addon_enable(module="better_fbx")
                bpy.ops.better_import.fbx(filepath=character_path, use_optimize_for_blender=False,
                                          use_auto_bone_orientation=True,
                                          use_reset_mesh_origin=True, use_reset_mesh_rotation=True,
                                          use_detect_deform_bone=True, use_auto_smooth=True,
                                          use_animation=True)
            except:
                bpy.ops.import_scene.fbx(
                    filepath=character_path, use_anim=True)

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
        addon_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../../addons/VRM_Addon_for_Blender-release.zip')
        print('addon_path ', addon_path)
        bpy.ops.preferences.addon_install(overwrite=True, target='DEFAULT', filepath=addon_path,
                                          filter_folder=True, filter_python=False, filter_glob="*.py;*.zip")
        bpy.ops.preferences.addon_enable(module="VRM-Addon-for-Blender-README")
        bpy.ops.wm.vrm_license_warning(
            filepath=character_path,
            license_confirmations=[{"name": "LicenseConfirmation0",
                                    "message": "This VRM is licensed by VRoid Hub License \"Alterations: No\".",
                                    "url": "https://hub.vroid.com/license?allowed_to_use_user=everyone&characterization_allowed_user=author&corporate_commercial_use=disallow&credit=necessary&modification=disallow&personal_commercial_use=disallow&redistribution=disallow&sexual_expression=disallow&version=1&violent_expression=disallow",
                                    "json_key": "otherPermissionUrl"}, {"name": "LicenseConfirmation1",
                                                                        "message": "This VRM is licensed by VRoid Hub License \"Alterations: No\".",
                                                                        "url": "https://hub.vroid.com/license?allowed_to_use_user=everyone&characterization_allowed_user=author&corporate_commercial_use=disallow&credit=necessary&modification=disallow&personal_commercial_use=disallow&redistribution=disallow&sexual_expression=disallow&version=1&violent_expression=disallow",
                                                                        "json_key": "otherLicenseUrl"}],
            import_anyway=True, extract_textures_into_folder=False, make_new_texture_folder=True)
    elif ".glb" == character_type:
        bpy.ops.import_scene.gltf(filepath=character_path)
    elif ".blend" == character_type:
        bpy.ops.wm.open_mainfile(filepath=character_path)
    else:
        print("data formate not support", character_type)
        return

    # Remove weapon mesh
    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    size_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            if any(wp in obj.name for wp in weapon_str_list):
                obj.select_set(state=True)
                bpy.ops.object.delete()
            else:
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
        print('posemode_toggle failed')

    obj_center, length, diagonal, max_point, min_point, mesh_verts = compute_mesh_size(
        meshes)
    print(obj_center, max_point, min_point)

    if not use_unit_transform:
        if not use_outside_transform:
            original_mesh = trimesh.base.Trimesh(vertices=mesh_verts)
            hull_vertices = original_mesh.convex_hull.vertices

            try:
                bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
                    hull_vertices)
            except:
                time.sleep(0.1)
                print("Miniball failed. Retry once...............")
                try:
                    bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
                        hull_vertices)
                except:
                    time.sleep(0.1)
                    print("Miniball failed. Retry second time...............")
                    try:
                        bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
                            hull_vertices)
                    except:
                        time.sleep(0.1)

            obj_center = bounding_sphere_C
            length = 2 * math.sqrt(bounding_sphere_r2)

        standard_height = 1.98  # meter
        for mesh in meshes:
            trn = -1 * obj_center[..., np.newaxis]
            print(transform_path)
            if use_outside_transform and len(transform_path) > 1 and os.path.exists(transform_path):
                T = np.loadtxt(transform_path)
                print("outside ", T)
            else:
                transform_path = os.path.join(dump_path, "transformation.txt")
                scale = standard_height / length
                T = np.eye(4)
                T[:3, 3:] = scale * trn
                T[:3, :3] = scale * T[:3, :3]
                print("inside ", T)
                np.savetxt(transform_path, T)

            mesh.matrix_world = Matrix(T) @ mesh.matrix_world
            # bpy.ops.object.transform_apply(scale=True)

    _, _, _, max_point, min_point, _ = compute_mesh_size(meshes)

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

    if colored_background:
        # override these options
        emission_render = False
        only_render_png = True

        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_settings.view_transform = 'Standard'
        composite_node_tree = bpy.context.scene.node_tree
        composite_alpha_node = composite_node_tree.nodes.new(
            "CompositorNodeAlphaOver")
        composite_alpha_node.premul = 1.0
        composite_alpha_node.inputs[1].default_value = (background_certain_color[0],
                                                        background_certain_color[1],
                                                        background_certain_color[2],
                                                        background_certain_color[3])
        for node in composite_node_tree.nodes:
            print(node.type)
            if node.type == "R_LAYERS":
                if len(node.outputs["Image"].links) > 0:
                    print(composite_alpha_node.inputs)
                    l = node.outputs["Image"].links[0]
                    final_output_node = l.to_node
                    if l is not None:
                        composite_node_tree.links.remove(l)
                    composite_node_tree.links.new(
                        node.outputs["Image"], composite_alpha_node.inputs[2])
                    composite_node_tree.links.new(
                        composite_alpha_node.outputs["Image"], final_output_node.inputs["Image"])

    scaled_mesh_filename = os.path.join(obj_dum_path, "scaled.obj")
    if export_scaled_obj:
        bpy.ops.wm.obj_export(filepath=scaled_mesh_filename)

    n_cam = len(cam_poses)
    camera_points_list = []
    render_camera_data = {}
    all_camera_names = list(cam_poses.keys())

    thumbnail_names = [100, 22, 121, 41, 68, 85, 91, 349, 322]
    vae_latent_names = [50, 53, 58, 61, 63, 66, 69, 71, 74,
                        79, 82, 84, 90, 95, 100, 105, 116, 129, 142, 171, 189]

    for index in range(len(all_camera_names)):
        camera_name = all_camera_names[index]
        if render_thumbnail:
            if index not in thumbnail_names:
                continue
        pose = np.array(cam_poses[camera_name]["pose"])
        K = np.array(cam_poses[camera_name]["k"])
        R_list = pose[0:3, 0:3].tolist()
        R = Matrix(R_list).to_euler('XYZ')
        t = Vector(pose[0:3, 3])
        camera_points_list.append(t)

        camera_angle = 2.0 * np.arctan2(K[0][2], K[0][0])
        pose_opencv = blender_to_opencv(pose)
        render_camera_data[camera_name] = {
            "k": np.asarray(K).tolist(),
            "pose": pose_opencv.tolist()
        }
        camera_data = bpy.data.cameras.new(name=camera_name)
        camera_object = bpy.data.objects.new(camera_name, camera_data)
        bpy.context.scene.collection.objects.link(camera_object)

        set_camera(camera_data, angle=camera_angle, W=W, H=H)

        camera_data.display_size = standard_height * 0.1
        camera_data.clip_start = 0.01
        camera_data.clip_end = 100
        camera_object.location = t
        camera_object.rotation_euler = R

        bpy.context.view_layer.update()  # update camera params

    camera_points_array = np.array(camera_points_list)
    store_pose_json_path = os.path.join(dump_path, "cam_parameters.json")
    if not no_camera_export:
        json_object = json.dumps(render_camera_data, indent=4)
        with open(store_pose_json_path, "w") as outfile:
            outfile.write(json_object)
        numpy_array_2_xyz(camera_points_array,
                          os.path.join(dump_path, "camera.xyz"))

    if not emission_render:
        if use_light == "point":
            lights_center = sphere_point_sample(n_lights)
            lights_center = lights_center * standard_height  # scale and translate
            for i in range(n_lights):
                bpy.ops.object.light_add(type='POINT',
                                         radius=np.random.normal(
                                             standard_height, standard_height * 0.1),
                                         align='WORLD',
                                         location=Vector(lights_center[i]),
                                         scale=(1, 1, 1))
                bpy.context.object.data.energy = light_energy
        elif use_light == "hdr":
            # load hdr env map
            hdr_file = hdr_map_path
            if len(hdr_file) < 1 or not os.path.exists(hdr_file):
                hdr_file = os.path.join(os.path.dirname(os.path.abspath(
                    __file__)), '../../irrmaps/aerodynamics_workshop_2k.hdr')
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
            environment_texture_node.image = hdr_image
            output_node = nodes.new("ShaderNodeOutputWorld")

            # connect the nodes
            links.new(
                environment_texture_node.outputs["Color"], output_node.inputs["Surface"])
            bpy.context.scene.render.film_transparent = True
        else:
            print('ERROR invalid light mode ', use_light)
            raise NotImplementedError

    if engine_type == "cycles":
        bpy.context.scene.render.engine = 'CYCLES'
    elif engine_type == "eevee":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    else:
        print("not support engine_type")
        return

    if emission_render:
        for mesh in meshes:
            change_shader_emission(mesh)

    for mesh in meshes:
        fix_bump_color_space(mesh)
        fix_material_space(mesh, "Metallic")
        fix_material_space(mesh, "Roughness")

    if engine_type == "cycles":
        if device == "GPU":
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                # only use GPU.
                if 'Intel' in d["name"] or 'AMD' in d["name"]:
                    d["use"] = 0
                else:
                    d["use"] = 1
                print(d["name"], ",", d["id"], ",", d["type"], ",", d["use"])

            bpy.context.scene.render.film_transparent = True
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            bpy.context.view_layer.use_pass_z = True
            bpy.context.view_layer.use_pass_normal = True
            if only_render_png:
                bpy.context.scene.render.image_settings.color_depth = '8'
                bpy.context.scene.render.image_settings.file_format = 'PNG'
            else:
                bpy.context.scene.render.image_settings.color_depth = '16'
                bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        else:
            bpy.context.scene.cycles.device = 'CPU'
            bpy.context.scene.render.film_transparent = True
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
            bpy.context.scene.render.image_settings.color_depth = '32'
            bpy.context.view_layer.use_pass_z = True
            bpy.context.view_layer.use_pass_normal = True
    else:
        # bpy.context.space_data.context = 'RENDER'
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.use_high_quality_normals = True
        if only_render_png:
            bpy.context.scene.render.image_settings.color_depth = '8'
            bpy.context.scene.render.image_settings.file_format = 'PNG'
        else:
            bpy.context.scene.render.image_settings.color_depth = '16'
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_normal = True

    if debug_blend_save:
        debug_blend_file = os.path.join(dump_path, "debug.blend")
        bpy.ops.wm.save_as_mainfile(filepath=debug_blend_file,
                                    compress=False,
                                    check_existing=False)

    # for mesh in meshes:
    #     set_default_value(mesh, 0.5, material_input_type="Metallic")
    #     set_default_value(mesh, 0.5, material_input_type="Roughness")

    if True:
        for index in range(len(all_camera_names)):
            camera_name = all_camera_names[index]
            if render_thumbnail:
                if index not in thumbnail_names:
                    continue
            bpy.context.scene.camera = bpy.data.objects[camera_name]

            if only_render_png:
                filepath = os.path.join(color_dump_path, str(
                    render_prefix) + str(camera_name) + ".png")
            else:
                filepath = os.path.join(exr_dump_path, str(
                    render_prefix) + str(camera_name) + ".exr")

            bpy.context.scene.render.filepath = filepath
            bpy.ops.render.render(write_still=True)

            if parse_exr:
                if not only_render_png:
                    exr_parser_script_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "../../exr_parser.py")
                    # if character_type == ".blend":
                    #     cmd = "python exr_parser.py --exr_file '{}' --blend".format(filepath)
                    # else:

                    cmd = "python {} --exr_file \"{}\" --camera_info_path \"{}\" --remove_exr ".format(
                        exr_parser_script_path, filepath, pose_json_path)
                    os.system(cmd)

    if render_thumbnail:
        if os.path.exists(color_dump_path):
            thumbnail_path = os.path.join(
                dump_path, render_prefix + "thumbnail.png")
            thumbnail_script_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "../../generate_thumbnail.py")
            cmd = "python {} --color_folder_path \"{}\" --save_thumbnail_path \"{}\" ".format(thumbnail_script_path,
                                                                                              color_dump_path,
                                                                                              thumbnail_path)
            os.system(cmd)

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

    parser = argparse.ArgumentParser(description='Render data script.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--output_folder', type=str,
                        default="", help='render result output folder')
    parser.add_argument('--transform_path', type=str, default="",
                        help='transform txt file path...')
    parser.add_argument('--pose_json_path', type=str,
                        help='transform txt file path...')
    parser.add_argument('--parse_exr', action='store_true',
                        help='parser exr in render process, by default do not parse')
    parser.add_argument('--render_daz', action='store_true',
                        help='render daz obj file which uses different coordinate system')
    parser.add_argument('--only_render_png', action='store_true',
                        help='only render png for visualization')
    parser.add_argument('--no_solidify', action='store_true',
                        help='do not use solidify to eliminate some artifacts')
    parser.add_argument('--smooth', action='store_true',
                        help='smooth mesh rendering images')
    parser.add_argument('--solidify_thickness', type=float,
                        default=0.005, help='thickness of solidify')
    parser.add_argument('--file_prefix', type=str, default="",
                        help='render results output file prefix; use this if you want unique marks on the render results')
    parser.add_argument('--engine', type=str, default="eevee",
                        help='render engine in blender, can be eevee or cycles')
    parser.add_argument('--render_height', type=int,
                        default=512, help='height of render output image')
    parser.add_argument('--render_width', type=int,
                        default=512, help='width of render output image')
    parser.add_argument('--render_lights_number', type=int, default=20,
                        help='number of lights used in rendering process; only valid if light_source="light"')
    parser.add_argument('--light_source', type=str, default="hdr",
                        help='light source used in rendering, "hdr" will use a pre-defined hdr map; "light" will use blender internal light source and light number is determined by render_lights_number')
    parser.add_argument('--hdr_map_path', type=str, default='irrmaps/aerodynamics_workshop_2k.hdr',
                        help='path of hdr light map to be used')
    parser.add_argument('--light_energy', type=float, default=1.0,
                        help='determines light energy (light brightness) used in rendering; only valid if light_source="light"')
    parser.add_argument('--render_thumbnail', action='store_true',
                        help='only render 9 views for thumbnail visualization')
    parser.add_argument('--use_outside_transform', action='store_true',
                        help='use outside transform txt file...')
    parser.add_argument('--use_unit_transform', action='store_true',
                        help='use unit transform inside, ignore all other transforms...')
    parser.add_argument('--no_camera_export', action='store_true',
                        help='no export camera parameters json and xyz file...')
    parser.add_argument('--use_better_fbx', action='store_true',
                        help='use very slow but more robust better fbx import plugin...')
    parser.add_argument('--export_scaled_obj', action='store_true',
                        help='export scaled mesh results in obj format...')
    parser.add_argument('--only_emission', action='store_true',
                        help='substitute pbr node to emission node to produce raw texture...')
    parser.add_argument('--colored_background', action='store_true',
                        help='force to use colored background in rendering output...')
    parser.add_argument('--debug_blend_save', action='store_true',
                        help='save blend file in render process for debug...')
    parser.add_argument('--background_certain_color', type=float, nargs=4, default=[1.0, 1.0, 1.0, 1.0],
                        help='color of the background when colored_background=True')

    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    pose_json_path = args.pose_json_path
    output_folder = args.output_folder
    parse_exr = args.parse_exr
    render_daz = args.render_daz
    only_render_png = args.only_render_png
    render_height = args.render_height
    render_width = args.render_width
    render_lights_number = args.render_lights_number
    file_prefix = args.file_prefix
    light_source = args.light_source
    hdr_map_path = args.hdr_map_path
    light_energy = args.light_energy
    engine_type = args.engine
    no_solidify = args.no_solidify
    solidify_thickness = args.solidify_thickness
    use_outside_transform = args.use_outside_transform
    use_unit_transform = args.use_unit_transform
    no_camera_export = args.no_camera_export
    transform_path = args.transform_path
    use_better_fbx = args.use_better_fbx
    export_scaled_obj = args.export_scaled_obj
    only_emission = args.only_emission
    colored_background = args.colored_background
    background_certain_color = args.background_certain_color
    debug_blend_save = args.debug_blend_save

    solidify = not no_solidify
    smooth = args.smooth

    render_thumbnail = args.render_thumbnail

    if len(output_folder) == 0:
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_folder = os.path.join(os.path.dirname(
            mesh_path), "{}_output_{}".format(name, render_height))

    weapon = ["weapon", "Weapon"]
    body = ["upperGarment"]

    oneExample_process(mesh_path, output_folder,
                       transform_path, pose_json_path,
                       render_height, render_width,
                       weapon_str_list=weapon,
                       only_render_png=only_render_png,
                       n_lights=render_lights_number,
                       render_prefix=file_prefix,
                       parse_exr=parse_exr,
                       render_daz=render_daz,
                       use_light=light_source,
                       hdr_map_path=hdr_map_path,
                       light_energy=light_energy,
                       use_solidify=solidify,
                       use_smooth=smooth,
                       solidify_thickness=solidify_thickness,
                       render_thumbnail=render_thumbnail,
                       use_outside_transform=use_outside_transform,
                       use_unit_transform=use_unit_transform,
                       no_camera_export=no_camera_export,
                       use_better_fbx=use_better_fbx,
                       export_scaled_obj=export_scaled_obj,
                       emission_render=only_emission,
                       colored_background=colored_background,
                       background_certain_color=background_certain_color,
                       debug_blend_save=debug_blend_save)

    time_point2 = time.time()
    print("Rendering mesh %s uses %f" %
          (mesh_path, (time_point2 - time_point1)))
    write_done(output_folder)
