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


def unwarpme_uv(mesh_1, unwarp_margin: float = 0.001):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_1.select_set(True)
    bpy.context.view_layer.objects.active = mesh_1
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.context.object.unwrapMeProps.startFromSelected = False
    bpy.context.object.unwrapMeProps.dPresets = '3'
    bpy.ops.uv.grow_charts()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=unwarp_margin)
    bpy.ops.uv.remove_overlaps()
    bpy.ops.object.editmode_toggle()


def remove_uv(mesh_1):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_1.select_set(True)
    bpy.context.view_layer.objects.active = mesh_1
    for uv in mesh_1.data.uv_layers:
        uv.active = True
        mesh_1.data.uv_layers.remove(uv)


def create_uv(mesh_1, new_uvmap_name="mesh_1_uv"):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_1.select_set(True)
    bpy.context.view_layer.objects.active = mesh_1
    new_uv = mesh_1.data.uv_layers.new(name=new_uvmap_name)
    # set the uv in obj_a as active
    new_uv.active = True


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


def add_image_texture_node(the_mesh,
                           mesh_folder,
                           image_width: int = 128,
                           image_height: int = 128,
                           material_name: str = ""):
    if the_mesh.material_slots:
        for slot in the_mesh.material_slots:
            node_tree = slot.material.node_tree
            for node in node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    print("No need to add material to this object....")
                    return the_mesh

    if the_mesh.material_slots:
        for i in range(len(the_mesh.material_slots)):
            with bpy.context.temp_override(object=the_mesh):
                bpy.ops.object.material_slot_remove()

    material_name = material_name.replace(" ", "_WSB_")
    create_material(object=the_mesh, material_name=material_name)
    diffuse_image = bpy.data.images.new(material_name + "_d", width=image_width, height=image_height)
    image_texture_name = material_name + "_remesh_d.png"
    image_texture_path = os.path.join(mesh_folder, image_texture_name)
    diffuse_image.save_render(image_texture_path)

    change_texture_image(the_mesh, image_texture_path)

    return the_mesh


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Baking start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='input mesh path')
    parser.add_argument('--output_mesh_path', type=str,
                        help='path to output mesh, with uv from source and geometry from destination')
    parser.add_argument('--debug_blend_save', action='store_true',
                        help='save blend file in process for debug...')
    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_path = args.mesh_path
    output_mesh_path = args.output_mesh_path
    debug_blend_save = args.debug_blend_save

    source_mesh_name = os.path.split(mesh_path)[1]
    source_mesh_filename = os.path.splitext(source_mesh_name)[0]
    output_mesh_folder = os.path.split(output_mesh_path)[0]
    output_texture_path = os.path.join(output_mesh_folder, "temp_texture.png")

    source_mesh_list = load_mesh(mesh_path)
    joint_source_mesh = join_list_of_mesh(source_mesh_list)

    remove_uv(joint_source_mesh)
    create_uv(joint_source_mesh)
    unwarpme_uv(joint_source_mesh)

    if debug_blend_save:
        debug_blend_file = os.path.join(output_mesh_folder, "debug.blend")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=debug_blend_file,
                                    compress=False,
                                    check_existing=False)

    add_image_texture_node(joint_source_mesh, output_mesh_folder,
                           material_name="temp_" + source_mesh_filename)

    export_mesh_obj(joint_source_mesh, output_mesh_path, path_mode='COPY')
