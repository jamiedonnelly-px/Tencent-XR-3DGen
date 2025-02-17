import argparse
import math
import os
import sys
import time

import bpy


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


def fix_bump_color_space(object):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name
            nodes = node_tree.nodes
            links = node_tree.links

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


def export_glb(mesh, mesh_path: str):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=mesh_path, export_format='GLB', use_selection=True)
    time.sleep(0.1)


def export_fbx(mesh, mesh_path: str):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.export_scene.fbx(
        filepath=mesh_path, use_selection=True, path_mode='COPY', embed_textures=True)
    time.sleep(0.1)


def copy_uv(mesh_1, mesh_2):
    uv_map_names = []
    new_uvmap_prefix = 'projected_uv_'
    # new_uv_map = mesh_1.uv_layers.new(name=new_uvmap_name)
    # new_uv_map.active = True

    mesh_1.select_set(True)
    mesh_2.select_set(True)
    bpy.context.view_layer.objects.active = mesh_2
    for uv in mesh_2.data.uv_layers:
        # set the uv in obj_b to active
        uv.active = True
        # create a new uv in obj_a
        new_uvmap_name = new_uvmap_prefix + uv.name
        uv_map_names.append(new_uvmap_name)
        new_uv = mesh_1.data.uv_layers.new(name=new_uvmap_name)
        # set the uv in obj_a as active
        new_uv.active = True
        bpy.ops.object.join_uvs()

    return uv_map_names


def check_shader_emission(object):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name
            nodes = node_tree.nodes
            links = node_tree.links

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if version_info[0] > 3:
                        if node.inputs["Emission Strength"].default_value > 0.0001:
                            new_emission_mat = create_material(
                                object, material_name=material_name + "_emission")
                            set_node_default_value(new_emission_mat,
                                                   node.inputs["Emission Strength"].default_value,
                                                   material_input_type="Emission Strength")
                            if len(node.inputs["Emission Color"].links) > 0:
                                l = node.inputs["Emission Color"].links[0]
                                if l.from_socket.name == 'Color':
                                    tex_image_node = l.from_node
                                    move_texture_node(
                                        new_emission_mat, tex_image_node, material_input_type="Emission Color")
                                    node_tree.links.remove(l)
                                    node_tree.nodes.remove(tex_image_node)
                            else:
                                set_node_default_value(new_emission_mat,
                                                       node.inputs["Emission Color"].default_value,
                                                       material_input_type="Emission Color")


def connect_color_to_emission(object):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name
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
                    time.sleep(0.1)
                    if version_info[0] > 3:
                        links.new(
                            color_attribute_node.outputs["Color"], node.inputs["Emission Color"])
                        node.inputs["Base Color"].default_value = (0, 0, 0, 1)
                        node.inputs["Emission Strength"].default_value = 1
                    else:
                        links.new(
                            color_attribute_node.outputs["Color"], node.inputs["Emission"])
                        node.inputs["Emission Strength"].default_value = 1
                        node.inputs["Base Color"].default_value = (0, 0, 0, 1)


def change_shader_emission(object):
    version_info = bpy.app.version
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            material_name = slot.material.name
            nodes = node_tree.nodes
            links = node_tree.links

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
                            node.inputs["Emission Strength"].default_value = 1
                        else:
                            node.inputs["Emission"].default_value = node.inputs['Base Color'].default_value


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


# color type: 'Non-Color' / 'sRGB'
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
                           image_width: int = 2048,
                           image_height: int = 2048,
                           material_name: str = ""):
    version_info = bpy.app.version
    print(version_info)

    if object.material_slots:
        for i in range(len(object.material_slots)):
            with bpy.context.temp_override(object=object):
                bpy.ops.object.material_slot_remove()

    create_material(object=object, material_name=material_name)
    diffuse_image = bpy.data.images.new(
        "bake_d", width=image_width, height=image_height)

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            diffusion_node = nodes.new("ShaderNodeTexImage")
            diffusion_node.image = diffuse_image

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(node.inputs)
                    if len(node.inputs["Base Color"].links) > 0:
                        l = node.inputs["Base Color"].links[0]
                        if l is not None:
                            node_tree.links.remove(l)

                    links.new(
                        diffusion_node.outputs["Color"], node.inputs["Base Color"])
                    links.new(
                        diffusion_node.outputs["Alpha"], node.inputs["Alpha"])

            nodes.active = diffusion_node
            diffusion_node.select = True
            node_tree.nodes.active = diffusion_node
            time.sleep(0.1)

    return diffuse_image


def add_emission_texture_node(object, image_width: int = 2048, image_height: int = 2048, material_name: str = ""):
    version_info = bpy.app.version
    if object.material_slots:
        for i in range(len(object.material_slots)):
            with bpy.context.temp_override(object=object):
                bpy.ops.object.material_slot_remove()

    create_material(object=object, material_name=material_name)
    diffuse_image = bpy.data.images.new(
        "bake_d", width=image_width, height=image_height)

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            diffusion_node = nodes.new("ShaderNodeTexImage")
            diffusion_node.image = diffuse_image

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(node.inputs)
                    if len(node.inputs["Emission Color"].links) > 0:
                        l = node.inputs["Emission Color"].links[0]
                        if l is not None:
                            node_tree.links.remove(l)
                    node.inputs["Base Color"].default_value = (0, 0, 0, 1)
                    node.inputs["Emission Strength"].default_value = 1
                    links.new(
                        diffusion_node.outputs["Color"], node.inputs["Emission Color"])
                    links.new(
                        diffusion_node.outputs["Alpha"], node.inputs["Alpha"])

            nodes.active = diffusion_node
            diffusion_node.select = True
            time.sleep(0.1)

    return diffuse_image


def add_uv_map_texture_node(object, uv_map_names: list):
    version_info = bpy.app.version
    print(version_info)

    if len(uv_map_names) == 0:
        return

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            uvmap_node = nodes.new("ShaderNodeUVMap")
            uvmap_node.uv_map = uv_map_names[0]

            # emission_node = nodes.new("ShaderNodeEmission")
            # emission_node.inputs['Strength'].default_value = 0.2
            for node in node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    links.new(uvmap_node.outputs["UV"], node.inputs["Vector"])


def remove_image_linkage(object, material_input_type: str = "Roughness"):
    version_info = bpy.app.version
    print(version_info)
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print(material_input_type, node.inputs)
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


def remove_default_value(object, material_input_type: str = "Emission Strength"):
    if object.material_slots:
        for slot in object.material_slots:
            set_node_default_value(slot.material, 0.0, material_input_type)


def mesh_shade_smooth(object):
    object.data.use_auto_smooth = True
    object.data.auto_smooth_angle = math.radians(30)
    for f in object.data.polygons:
        f.use_smooth = True


# bake diffuse from mesh_1 to mesh_2
def bake_diffuse(mesh_1, mesh_2, device: str,
                 texture_folder: str, image_width: int = 2048, image_height: int = 2048,
                 cage_extrusion: float = 0.001, max_ray_distance: float = 0.01):
    texture_image_path = os.path.join(texture_folder, "atlas_d.png")
    texture_image_extension = os.path.splitext(texture_image_path)[1]
    texture_image_extension = texture_image_extension.replace(".", "")
    texture_image_type = texture_image_extension.upper()

    bpy.context.scene.view_settings.view_transform = 'Standard'
    diffuse_image = add_emission_texture_node(
        mesh_2, image_width=image_width, image_height=image_height, material_name="bake_mat")

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 8
    if device == 'GPU':
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for current_device in bpy.context.preferences.addons["cycles"].preferences.devices:
            print(current_device["name"])
            # only use NVIDIA GPU.
            if 'Intel' in current_device["name"] or 'AMD' in current_device["name"]:
                current_device["use"] = 0
            else:
                current_device["use"] = 1
    else:
        bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.use_preview_denoising = True
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.cage_extrusion = cage_extrusion
    bpy.context.scene.render.bake.max_ray_distance = max_ray_distance
    bpy.context.scene.cycles.bake_type = 'EMIT'

    bpy.ops.object.select_all(action='DESELECT')
    mesh_2.select_set(True)
    mesh_1.select_set(True)
    bpy.context.view_layer.objects.active = mesh_2
    bpy.ops.object.bake('INVOKE_DEFAULT', type='EMIT', save_mode='EXTERNAL')

    diffuse_image.save_render(texture_image_path)
    remove_image_linkage(mesh_2, "Base Color")
    if bpy.app.version[0] > 3:
        remove_image_linkage(mesh_2, "Emission Color")
        remove_default_value(mesh_2, "Emission Strength")
        remove_default_value(mesh_2, "Specular IOR Level")
    else:
        remove_image_linkage(mesh_2, "Emission")
        remove_default_value(mesh_2, "Emission Strength")
        remove_default_value(mesh_2, "Specular")
    change_texture_image(mesh_2, texture_image_path)


def duplicate(object, copy_data=True, copy_actions=False):
    obj_copy = object.copy()
    if copy_data:
        obj_copy.data = obj_copy.data.copy()
    if copy_actions and obj_copy.animation_data:
        obj_copy.animation_data.action = obj_copy.animation_data.action.copy()
    return obj_copy


def remove_uv(mesh_1):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_1.select_set(True)
    bpy.context.view_layer.objects.active = mesh_1
    for uv in mesh_1.data.uv_layers:
        uv.active = True
        mesh_1.data.uv_layers.remove(uv)


def smart_uv(the_mesh, angle_limit=1.04):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    # lm = the_mesh.data.uv_layers.get("MyUV")
    # if not lm:
    #     lm = the_mesh.data.uv_layers.new(name="MyUV")
    # lm.active = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=0.873, margin_method='ADD')
    bpy.ops.object.mode_set(mode='OBJECT')
    return the_mesh


def decimate(the_mesh, target_faces_num):
    num_faces = len(the_mesh.data.polygons)
    if num_faces > target_faces_num:
        decimate_ratio = target_faces_num / num_faces
        decimator = the_mesh.modifiers.new("decimate", "DECIMATE")
        decimator.ratio = decimate_ratio
        bpy.ops.object.convert(target='MESH')
    return the_mesh


def remove_doubles(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.object.mode_set(mode='OBJECT')
    time.sleep(0.1)


def remesh_quad(the_mesh, target_face_number=4000):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    bpy.context.object.data.remesh_mode = 'QUAD'
    bpy.ops.object.quadriflow_remesh(target_faces=target_face_number)
    return the_mesh


def triangulate(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    the_mesh.modifiers.new("triangulate", "TRIANGULATE")
    bpy.ops.object.convert(target='MESH')  # bake modifier to mesh
    return the_mesh


def main_bake_verts2tex(source_mesh_path, output_mesh_folder, output_mesh_filename,
                        rebuild_source_material=True, source_color_attribute=True, device="CPU",
                        save_debug_blend_file=False, texture_image_width=2048, texture_image_height=2048,
                        cage_extrusion=0.02, max_ray_distance=0.04, use_quad_remesh=False,
                        decimate_face_num=12000, quad_remesh_face_num=6000):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    os.makedirs(output_mesh_folder, exist_ok=True)

    source_mesh_list = load_mesh(source_mesh_path)
    source_mesh = join_list_of_mesh(source_mesh_list)

    if rebuild_source_material:
        create_material(object=source_mesh, material_name="new_material")
        if bpy.app.version[0] > 3:
            remove_default_value(source_mesh, material_input_type="Specular IOR Level")
        else:
            remove_default_value(source_mesh, material_input_type="Specular")

    destination_mesh = duplicate(source_mesh)
    destination_mesh.name = "wsb"
    bpy.context.collection.objects.link(destination_mesh)

    destination_mesh = decimate(destination_mesh, target_faces_num=decimate_face_num)
    if use_quad_remesh:
        destination_mesh = remesh_quad(destination_mesh, target_face_number=quad_remesh_face_num)

    remove_uv(destination_mesh)
    smart_uv(destination_mesh)

    fix_bump_color_space(source_mesh)
    # fix_material_space(source_mesh, "Metallic")
    # fix_material_space(source_mesh, "Roughness")
    mesh_shade_smooth(source_mesh)
    remove_image_linkage(source_mesh, "Metallic")
    remove_image_linkage(source_mesh, "Roughness")

    check_shader_emission(source_mesh)
    if source_color_attribute:
        connect_color_to_emission(source_mesh)
    else:
        change_shader_emission(source_mesh)

    uncompressed_blend_path = os.path.join(output_mesh_folder, "bake.blend")
    if len(output_mesh_filename) > 1:
        output_mesh_path = os.path.join(output_mesh_folder, output_mesh_filename + ".obj")
        output_fbx_path = os.path.join(output_mesh_folder, output_mesh_filename + ".fbx")
        output_glb_path = os.path.join(output_mesh_folder, output_mesh_filename + ".glb")
    else:
        output_mesh_path = os.path.join(output_mesh_folder, "bake.obj")
        output_fbx_path = os.path.join(output_mesh_folder, "bake.fbx")
        output_glb_path = os.path.join(output_mesh_folder, "bake.glb")

    bake_diffuse(source_mesh,
                 destination_mesh,
                 device=device,
                 texture_folder=output_mesh_folder,
                 image_width=texture_image_width,
                 image_height=texture_image_height,
                 cage_extrusion=cage_extrusion,
                 max_ray_distance=max_ray_distance)
    if use_quad_remesh:
        destination_mesh = triangulate(destination_mesh)
    remove_doubles(destination_mesh)
    mesh_shade_smooth(destination_mesh)

    if destination_mesh.data.color_attributes:
        attrs = destination_mesh.data.color_attributes
        for r in range(len(destination_mesh.data.color_attributes) - 1, -1, -1):
            attrs.remove(attrs[r])

    if save_debug_blend_file:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=uncompressed_blend_path, compress=False, check_existing=False)
    time.sleep(0.1)

    export_mesh_obj(destination_mesh,
                    mesh_path=output_mesh_path,
                    path_mode="AUTO")
    export_fbx(destination_mesh, output_fbx_path)
    destination_mesh.data.use_auto_smooth = False
    export_glb(destination_mesh, output_glb_path)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)
    print("Baking end. Local time is %s" % (end_time_str))

    return True, output_mesh_path


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Baking start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path to source mesh')
    parser.add_argument('--rebuild_source_material',
                        action='store_true', help='rebuild material of source meshes')
    parser.add_argument('--source_color_attribute',
                        action='store_true',
                        help='use color attributes and ignore all texture images')
    parser.add_argument('--device', type=str, help='baking device, CPU or GPU')
    parser.add_argument('--save_debug_blend_file',
                        action='store_true', help='save blend file for debug')
    parser.add_argument('--output_mesh_folder', type=str,
                        help='folder of final output')
    parser.add_argument('--output_mesh_filename',
                        type=str,
                        default="",
                        help='expected file base name; do not use full path here, do not use extenstion here.')
    parser.add_argument('--texture_image_width', type=int,
                        default=2048, help='width of baked texture image')
    parser.add_argument('--texture_image_height', type=int,
                        default=2048, help='height of baked texture image')
    parser.add_argument('--cage_extrusion',
                        type=float,
                        default=0.02,
                        help='inflate object by this distance which helps matching to destination')
    parser.add_argument('--max_ray_distance',
                        type=float,
                        default=0.04,
                        help='may ray cast distance between source and destination objects')
    parser.add_argument('--use_quad_remesh',
                        action='store_true',
                        help='use quad remesh in mesh preprocessing')
    parser.add_argument('--decimate_face_num',
                        type=int,
                        default=12000,
                        help='face num after decimate')
    parser.add_argument('--quad_remesh_face_num',
                        type=int,
                        default=6000,
                        help='face number after quad remesh')
    args = parser.parse_args(raw_argv)

    source_mesh_path = args.source_mesh_path
    rebuild_source_material = args.rebuild_source_material
    source_color_attribute = args.source_color_attribute
    device = args.device
    save_debug_blend_file = args.save_debug_blend_file
    output_mesh_folder = args.output_mesh_folder
    output_mesh_filename = args.output_mesh_filename
    texture_image_width = args.texture_image_width
    texture_image_height = args.texture_image_height
    cage_extrusion = args.cage_extrusion
    max_ray_distance = args.max_ray_distance
    use_quad_remesh = args.use_quad_remesh
    decimate_face_num = args.decimate_face_num
    quad_remesh_face_num = args.quad_remesh_face_num

    suc_flag, output_mesh_path = main_bake_verts2tex(
        source_mesh_path,
        output_mesh_folder,
        output_mesh_filename,
        rebuild_source_material=rebuild_source_material,
        source_color_attribute=source_color_attribute,
        device=device,
        save_debug_blend_file=save_debug_blend_file,
        texture_image_width=texture_image_width,
        texture_image_height=texture_image_height,
        cage_extrusion=cage_extrusion,
        max_ray_distance=max_ray_distance,
        use_quad_remesh=use_quad_remesh,
        decimate_face_num=decimate_face_num,
        quad_remesh_face_num=quad_remesh_face_num
    )
