import bpy
import os
import sys
import time
import shutil
import argparse
import json
from PIL import Image


def find_texture_file(ColorTexture_Name, work_path):
    for filename in os.listdir(work_path):
        if filename.startswith(ColorTexture_Name):
            return filename
    return None


def save_separated_channels(image_path, output_dir, output_format='png'):
    """
    分离图像的RGB通道并保存为指定格式。

    参数:
    image_path (str): 原始多通道贴图的路径。
    output_dir (str): 输出目录的路径。
    output_format (str): 输出图像的格式（默认为'png'）。
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    img = Image.open(image_path)
    if img.mode == 'RGB':
        r, g, b = img.split()
    elif img.mode == 'RGBA':
        r, g, b, a = img.split()
    else:
        print(f"Unsupported image mode: {img.mode}. Exiting function.")
        return

    base_name = os.path.basename(image_path)
    file_name, _ = os.path.splitext(base_name)

    red_filename = f"{file_name}_R.{output_format}"
    red_filepath = os.path.join(output_dir, red_filename)
    r.save(red_filepath, output_format.upper())

    green_filename = f"{file_name}_G.{output_format}"
    green_filepath = os.path.join(output_dir, green_filename)
    g.save(green_filepath, output_format.upper())

    blue_filename = f"{file_name}_B.{output_format}"
    blue_filepath = os.path.join(output_dir, blue_filename)
    b.save(blue_filepath, output_format.upper())

    return red_filename, green_filename, blue_filename


def glb_convert(mesh_path: str, output_folder: str):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    bpy.ops.import_scene.gltf(filepath=mesh_path)
    imported_objects = bpy.context.selected_objects
    gltf_path = os.path.join(output_folder, mesh_basename + ".gltf")
    bpy.ops.export_scene.gltf(filepath=gltf_path, use_selection=True,
                              export_format='GLTF_SEPARATE', export_animations=False)

    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.import_scene.gltf(filepath=gltf_path)

    imported_objects = bpy.context.selected_objects
    obj_path = os.path.join(output_folder, mesh_basename + ".fbx")
    bpy.ops.export_scene.fbx(filepath=obj_path, use_selection=True, bake_anim=False)

    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.import_scene.fbx(filepath=obj_path)
    all_objects_materials_white = True

    for obj in bpy.context.scene.objects:

        material = obj.active_material
        if material and material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    if node.inputs['Base Color'].is_linked:
                        all_objects_materials_white = False
                        break
                    else:
                        color = node.inputs['Base Color'].default_value
                        if color[0] != color[1] or color[0] != color[2]:
                            all_objects_materials_white = False
                            break
            if not all_objects_materials_white:
                break

    time.sleep(0.1)


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


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, z_up=False):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    if z_up:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='Y', up_axis='Z',
                              global_scale=global_scale)
    else:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='NEGATIVE_Z', up_axis='Y',
                              global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


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

                    links.new(diffusion_node.outputs["Color"], node.inputs[material_input_type])
                    if material_input_type == 'Base Color':
                        links.new(diffusion_node.outputs["Alpha"], node.inputs["Alpha"])


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

    change_texture_image(object, image_texture_path)

    return object

def normalize_mesh_to_unit_range(object):
    from mathutils import Vector

    obj = object

    #Eventually apply transforms (comment if unwanted)
    bpy.ops.object.transform_apply( rotation = True, scale = True )

    minX = min( [vertex.co[0] for vertex in obj.data.vertices] )
    minY = min( [vertex.co[1] for vertex in obj.data.vertices] )
    minZ = min( [vertex.co[2] for vertex in obj.data.vertices] )
    maxX = max( [vertex.co[0] for vertex in obj.data.vertices] )
    maxY = max( [vertex.co[1] for vertex in obj.data.vertices] )
    maxZ = max( [vertex.co[2] for vertex in obj.data.vertices] )

    vMin = Vector( [minX, minY, minZ] )
    vMax = Vector( [maxX, maxY, maxZ] )

    min_coords = [min(vertex.co[i] for vertex in obj.data.vertices) for i in range(3)]
    max_coords = [max(vertex.co[i] for vertex in obj.data.vertices) for i in range(3)]

    ranges = [max_coord - min_coord for max_coord, min_coord in zip(max_coords, min_coords)]

    max_range = max(ranges)

    center = [(min_coords[i] + max_coords[i]) / 2.0 for i in range(3)]

    for v in obj.data.vertices:
        for i in range(3):
            v.co[i] = (v.co[i] - center[i]) / max_range * 2

    
if __name__ == '__main__':

    t_start = time.time()
    local_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("GLB to obj start. Local time is %s" % (start_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='glb to obj data script.')
    parser.add_argument('--mesh_path', type=str,
                        help='path of mesh glb file')
    parser.add_argument('--output_mesh_path', type=str,
                        help='output path of generated obj')
    parser.add_argument('--force_better_fbx', action='store_true',
                        help='force to use better fbx as import plugin; the plugin has to be pre-installed')
    parser.add_argument('--force_z_up', action='store_true',
                        help='force use z/y axis in obj exporting')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')
    parser.add_argument('--mesh_normalization', action='store_true',
                        help='normalization to [-1, 1]')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_mesh_path = args.output_mesh_path
    output_mesh_folder = os.path.split(output_mesh_path)[0]
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_extension = os.path.splitext(mesh_filename)[1].lower()
    mesh_extension_pure_name = mesh_extension.replace(".", "")

    os.makedirs(output_mesh_folder, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    if mesh_extension == '.fbx':
        if args.force_better_fbx:
            bpy.ops.preferences.addon_enable(module="better_fbx")
            bpy.ops.better_import.fbx(filepath=mesh_path, use_optimize_for_blender=False,
                                      use_auto_bone_orientation=True,
                                      use_reset_mesh_origin=True, use_reset_mesh_rotation=True,
                                      use_detect_deform_bone=True, use_auto_smooth=True,
                                      use_animation=True)

        else:
            bpy.ops.import_scene.fbx(filepath=mesh_path, use_anim=True)
    elif mesh_extension == '.glb':
        glb_convert(mesh_path=mesh_path, output_folder=output_mesh_folder)
    elif mesh_extension == '.obj':
        bpy.ops.wm.obj_import(filepath=mesh_path)

    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    size_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            meshes.append(obj)

    if len(meshes) < 1:
        print("No avatar found in model.....")
        exit(-1)

    joint_mesh = join_list_of_mesh(meshes)

    if len(joint_mesh.data.uv_layers) > 0:
        if not joint_mesh.material_slots:
            add_image_texture_node(joint_mesh, output_mesh_folder)
        else:
            if len(joint_mesh.material_slots) > 0:
                add_image_texture_node(joint_mesh, output_mesh_folder)

    print("Export %s mesh with from %s to %s" % (mesh_extension_pure_name, mesh_path, output_mesh_path))
    if args.mesh_normalization:
        print('mesh_normalization')
        normalize_mesh_to_unit_range(joint_mesh)
        print('mesh_normalization done')
        
    if args.copy_texture:
        export_mesh_obj(joint_mesh, output_mesh_path, path_mode='COPY', z_up=args.force_z_up)
    else:
        export_mesh_obj(joint_mesh, output_mesh_path, path_mode='RELATIVE', z_up=args.force_z_up)