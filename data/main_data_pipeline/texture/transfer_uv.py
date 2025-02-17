import argparse
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


def remove_uv(mesh_1):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_1.select_set(True)
    bpy.context.view_layer.objects.active = mesh_1
    for uv in mesh_1.data.uv_layers:
        uv.active = True
        mesh_1.data.uv_layers.remove(uv)


def copy_uv(mesh_1, mesh_2):
    bpy.ops.object.select_all(action='DESELECT')
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


def add_image_texture_node(object,
                           new_texture_path: str,
                           material_name: str = ""):
    version_info = bpy.app.version
    print(version_info)

    texture_filename = os.path.split(new_texture_path)[1]
    bpy.data.images.load(new_texture_path)

    if object.material_slots:
        for i in range(len(object.material_slots)):
            with bpy.context.temp_override(object=object):
                bpy.ops.object.material_slot_remove()

    create_material(object=object, material_name=material_name)

    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            diffusion_node = nodes.new("ShaderNodeTexImage")
            diffusion_node.image = bpy.data.images[texture_filename + '.001']

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


def get_image_texture_path(object):
    texture_image_path = ""
    if object.material_slots:
        for slot in object.material_slots:
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'TEX_IMAGE':
                    texture_image_path = node.image.filepath
                    return texture_image_path


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
    parser.add_argument('--destination_mesh_path', type=str,
                        help='path to destination mesh, transfer uv from source to destination')
    parser.add_argument('--output_mesh_path', type=str,
                        help='path to output mesh, with uv from source and geometry from destination')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    source_mesh_path = args.source_mesh_path
    destination_mesh_path = args.destination_mesh_path
    output_mesh_path = args.output_mesh_path

    source_mesh_name = os.path.split(source_mesh_path)[1]
    source_mesh_filename = os.path.split(source_mesh_name)[0]

    source_mesh_list = load_mesh(source_mesh_path)
    joint_source_mesh = join_list_of_mesh(source_mesh_list)
    source_texture_path = get_image_texture_path(joint_source_mesh)

    destination_mesh_list = load_mesh(destination_mesh_path)
    joint_destination_mesh = join_list_of_mesh(destination_mesh_list)

    remove_uv(joint_destination_mesh)
    copy_uv(joint_destination_mesh, joint_source_mesh)

    add_image_texture_node(joint_destination_mesh, source_texture_path,
                           material_name="transfer_" + source_mesh_filename)

    export_mesh_obj(joint_destination_mesh, output_mesh_path, path_mode='COPY')
