import argparse
import os
import sys
import time

import bpy
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


def glb_convert(mesh_path: str):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    bpy.context.scene.view_settings.view_transform = 'Standard'

    try:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
        imported_objects = bpy.context.selected_objects
        all_objects_materials_white = True
        has_vertex_color = False
        hsa_diff_basecolor = False
        Cnum = 0

        for material in bpy.data.materials:
            if material.users == 0:
                bpy.data.materials.remove(material)

        for obj in bpy.context.scene.objects:
            if obj.material_slots:
                for mat_slot in obj.material_slots:
                    mat = mat_slot.material
                    if mat:
                        if not mat.get("tex_process"):
                            mat["mat_process"] = False
                            mat["tex_process"] = False

        for mat in bpy.data.materials:
            print(
                f"Material: {mat.name}, mat_process: {mat.get('mat_process')}")

        for obj in bpy.context.scene.objects:
            # 检查对象是否有材质
            if obj.material_slots:
                for mat_slot in obj.material_slots:
                    mat = mat_slot.material
                    # 检查材质是否存在，是否使用节点，以及是否已经处理过
                    if mat and mat.use_nodes and not mat.get("tex_process"):
                        # 遍历材质中的所有节点
                        for node in mat.node_tree.nodes:
                            # 检查节点是否为图像纹理节点
                            if node.type == 'TEX_IMAGE':
                                image = node.image
                                if image:
                                    # 构建导出的图像文件路径
                                    file_path = os.path.join(mesh_folder, image.name.split(".")[0] + '.png')
                                    # 保存图像
                                    image.save_render(filepath=file_path)
                                    print(f"Saved: {file_path}")
                                else:
                                    print(
                                        f"No image found in node: {node.name}")
                        # 设置材质的 'tex_process' 属性为 True，表示已处理
                        mat["tex_process"] = True

        materials_dict = {}
        for mat in bpy.data.materials:
            mat_properties = {
                'basecolorR': None,
                'basecolorG': None,
                'basecolorB': None,
                'linkAlpha': False,
                'hasVertex': False,
                'colorTexture': None,
                'normalTexture': None,
                'normalStrength': None,
                'mtlTexture': None,
                'rouTexture': None,
                'aoTexture': None,
                'blenderMode': None,
                'hasEmission': False,
                'EmissionR': None,
                'EmissionG': None,
                'EmissionB': None,
            }

            materials_dict[mat.name] = mat_properties

        for obj in bpy.context.scene.objects:
            if not obj.material_slots:
                continue
            if obj.material_slots is None:
                continue

            for mat_slot in obj.material_slots:
                mat = mat_slot.material
                if mat is not None and not mat.get("mat_process", False):
                    mat["mat_process"] = True
                    mat_name = mat.name
                    materials_dict[mat.name] = mat_properties.copy()
                    blend_mode = mat.blend_method
                    materials_dict[mat.name]['blenderMode'] = blend_mode

                    if not mat.node_tree:
                        continue
                    if mat.node_tree is None:
                        continue

                    color = (0.8, 0.8, 0.8, 1.0)
                    Emissioncolor = (0.0, 0.0, 0.0, 1.0)
                    bool_diffuseBSDF = False
                    bool_BSDF = False
                    bool_ColorMix = False
                    bool_Emission = False

                    for node in mat.node_tree.nodes:
                        if node.bl_idname == 'ShaderNodeTexImage':
                            if node.label in ('BASE COLOR', 'DIFFUSE COLOR', 'DIFFUSE'):
                                color_texture_node = node
                                if color_texture_node.image and color_texture_node.image.name:
                                    ColorTex_Name = color_texture_node.image.name.split(".")[
                                        0]
                                    final_color_name = find_texture_file(
                                        ColorTex_Name, mesh_folder)
                                    all_objects_materials_white = False
                                    materials_dict[mat.name]['colorTexture'] = final_color_name
                                    if color_texture_node.outputs['Alpha'].is_linked:
                                        materials_dict[mat.name]['linkAlpha'] = True

                            if node.label == 'NORMALMAP':
                                normal_texture_node = node
                                if len(normal_texture_node.outputs['Color'].links) > 0:
                                    normal_vector_node = normal_texture_node.outputs['Color'].links[0].to_node
                                    normal_strength = normal_vector_node.inputs["Strength"].default_value
                                    materials_dict[mat.name]['normalStrength'] = normal_strength

                                NormalTex_Name = normal_texture_node.image.name.split(".")[
                                    0]
                                final_normal_name = find_texture_file(
                                    NormalTex_Name, mesh_folder)
                                materials_dict[mat.name]['normalTexture'] = final_normal_name

                            if node.label == 'METALLIC ROUGHNESS':
                                multi_texture_node = node
                                multiTex_Name = multi_texture_node.image.name.split(".")[
                                    0]
                                final_multi_name = find_texture_file(
                                    multiTex_Name, mesh_folder)
                                multi_tex_fullpath = os.path.join(
                                    mesh_folder, final_multi_name)
                                # output_dir = mesh_folder
                                red_filepath, green_filepath, blue_filepath = save_separated_channels(
                                    multi_tex_fullpath, mesh_folder, output_format='png')
                                if len(multi_texture_node.outputs['Color'].links) < 0:
                                    continue
                                color_seperate_node = multi_texture_node.outputs[
                                    'Color'].links[0].to_node
                                if len(color_seperate_node.outputs["Red"].links) > 0:
                                    print(
                                        color_seperate_node.outputs["Red"].links[0].to_socket.name)
                                    if color_seperate_node.outputs["Red"].links[0].to_socket.name == "Metallic":
                                        materials_dict[mat.name]['mtlTexture'] = red_filepath
                                    elif color_seperate_node.outputs["Red"].links[0].to_socket.name == "Roughness":
                                        materials_dict[mat.name]['rouTexture'] = red_filepath

                                if len(color_seperate_node.outputs["Green"].links) > 0:
                                    color_seperate_node.outputs["Green"].links[0].to_socket.name
                                    if color_seperate_node.outputs["Green"].links[0].to_socket.name == "Metallic":
                                        materials_dict[mat.name]['mtlTexture'] = green_filepath
                                    elif color_seperate_node.outputs["Green"].links[0].to_socket.name == "Roughness":
                                        materials_dict[mat.name]['rouTexture'] = green_filepath

                                if len(color_seperate_node.outputs["Blue"].links) > 0:
                                    color_seperate_node.outputs["Blue"].links[0].to_socket.name
                                    if color_seperate_node.outputs["Blue"].links[0].to_socket.name == "Metallic":
                                        materials_dict[mat.name]['mtlTexture'] = blue_filepath
                                    elif color_seperate_node.outputs["Blue"].links[0].to_socket.name == "Roughness":
                                        materials_dict[mat.name]['rouTexture'] = blue_filepath

                            for output in node.outputs:
                                for link in output.links:
                                    mat.node_tree.links.remove(link)

                        if node.bl_idname == 'ShaderNodeBsdfPrincipled':
                            BSDFcolor = node.inputs['Base Color'].default_value
                            bool_BSDF = True
                            for output in node.outputs:
                                for link in output.links:
                                    mat.node_tree.links.remove(link)

                        if node.bl_idname == 'ShaderNodeVertexColor':
                            all_objects_materials_white = False
                            has_vertex_color = True
                            materials_dict[mat.name]['hasVertex'] = True

                        if node.bl_idname == 'ShaderNodeBsdfDiffuse':
                            diffusecolor = node.inputs['Color'].default_value
                            bool_diffuseBSDF = True

                        if node.bl_idname == 'ShaderNodeMix':
                            if node.label == 'Color Factor':
                                factorcolor = node.inputs['B'].default_value
                                bool_ColorMix = True

                        if node.bl_idname == 'ShaderNodeEmission':
                            if node.inputs['Color'].is_linked:
                                print("Color input is connected.")
                            else:
                                Emissioncolor = node.inputs['Color'].default_value
                                bool_Emission = True

                    if bool_BSDF:
                        color = BSDFcolor

                    if bool_Emission:
                        color = Emissioncolor
                        materials_dict[mat.name]['hasEmission'] = True

                    if not bool_Emission:
                        materials_dict[mat.name]['hasEmission'] = False

                    if bool_diffuseBSDF:
                        color = diffusecolor

                    if bool_ColorMix:
                        if isinstance(factorcolor, (list, tuple)) and len(factorcolor) >= 3:
                            color = factorcolor
                        else:
                            color = (factorcolor, factorcolor,
                                     factorcolor, 1)
                        Mixfacotor = Mixfacotor + 1

                    materials_dict[mat.name]['basecolorR'] = color[0]
                    materials_dict[mat.name]['basecolorG'] = color[1]
                    materials_dict[mat.name]['basecolorB'] = color[2]

                    materials_dict[mat.name]['EmissionR'] = Emissioncolor[0]
                    materials_dict[mat.name]['EmissionG'] = Emissioncolor[1]
                    materials_dict[mat.name]['EmissionB'] = Emissioncolor[2]
                    if Emissioncolor[0] != Emissioncolor[1] or Emissioncolor[0] != Emissioncolor[2]:
                        all_objects_materials_white = False

                    if color[0] != color[1] or color[0] != color[2]:
                        all_objects_materials_white = False
                        hsa_diff_basecolor = True

                    Cnum = Cnum + 1

                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)

        for material in bpy.data.materials:
            if material.users == 0:
                bpy.data.materials.remove(material)

        output_filename_diffbaseClr = "diff_" if hsa_diff_basecolor else ""
        output_filename_vertex = "vertex_" if has_vertex_color else ""
        output_filename = "white_" if all_objects_materials_white else ""
        output_filename = output_filename + output_filename_vertex + output_filename_diffbaseClr

        imported_objects = bpy.context.selected_objects

        obj_path = os.path.join(mesh_folder, output_filename + mesh_basename + ".obj")
        bpy.ops.wm.obj_export(filepath=obj_path, export_materials=True, export_colors=True)

        # MTL 操作
        mtl_file_path = os.path.join(mesh_folder, output_filename + mesh_basename + ".mtl")

        with open(mtl_file_path, 'r') as mtl_file:
            lines = mtl_file.readlines()

        for i, line in enumerate(lines):
            if not line.startswith('newmtl '):
                continue
            material_name = line.split(' ')[1].strip()
            if material_name in materials_dict:
                end_of_material = i + 1
                kd_pos = i + 1
                while end_of_material < len(lines) and not lines[end_of_material].startswith('newmtl '):
                    end_of_material += 1
                while kd_pos < len(lines) and not lines[kd_pos].startswith('newmtl '):
                    if lines[kd_pos].startswith('Kd '):
                        break
                    kd_pos += 1

                color_texture = materials_dict[material_name].get(
                    'colorTexture')
                basecolor_R = materials_dict[material_name].get('basecolorR')
                basecolor_G = materials_dict[material_name].get('basecolorG')
                basecolor_B = materials_dict[material_name].get('basecolorB')
                link_alpha = materials_dict[material_name].get('linkAlpha')
                blend_Mode = materials_dict[material_name].get('blenderMode')
                has_vertex = materials_dict[material_name].get('hasVertex')
                normal_texture = materials_dict[material_name].get('normalTexture')
                normal_strength = materials_dict[material_name].get('normalStrength')
                rou_texture = materials_dict[material_name].get('rouTexture')
                mtl_texture = materials_dict[material_name].get('mtlTexture')
                # ao_texture = materials_dict[material_name].get(
                #     'aoTexture')

                has_emission = materials_dict[material_name].get('hasEmission')
                emission_R = materials_dict[material_name].get('EmissionR')
                emission_G = materials_dict[material_name].get('EmissionG')
                emission_B = materials_dict[material_name].get('EmissionB')

                defalut_color_value = 0.8
                defalut_emiss_value = 0

                basecolor_R = defalut_color_value if basecolor_R is None else basecolor_R
                basecolor_G = defalut_color_value if basecolor_G is None else basecolor_G
                basecolor_B = defalut_color_value if basecolor_B is None else basecolor_B

                emission_R = defalut_emiss_value if emission_R is None else emission_R
                emission_G = defalut_emiss_value if emission_G is None else emission_G
                emission_B = defalut_emiss_value if emission_B is None else emission_B
                kdparm = str(round(basecolor_R, 3)) + " " + str(round(basecolor_G, 3)) + " " + str(
                    round(basecolor_B, 3))
                emissionParm = str(round(emission_R, 3)) + " " + str(round(emission_G, 3)) + " " + str(
                    round(emission_B, 3))

                if color_texture is not None:
                    lines.insert(end_of_material, f'map_Kd {color_texture}\n')
                if normal_texture is not None and normal_strength is not None:
                    lines.insert(end_of_material, f'map_Bump -bm {normal_strength} {normal_texture}\n')
                if mtl_texture is not None:
                    lines.insert(end_of_material, f'map_refl {mtl_texture}\n')
                if rou_texture is not None:
                    lines.insert(end_of_material, f'map_Ns {rou_texture}\n')
                if link_alpha is not None:
                    lines.insert(end_of_material, f'link_Alpha {link_alpha}\n')
                if has_vertex is not None:
                    lines.insert(end_of_material, f'has_Vertex {has_vertex}\n')
                if blend_Mode is not None:
                    lines.insert(end_of_material, f'blend_Mode {blend_Mode}\n')
                if basecolor_R is not None or basecolor_G is not None or basecolor_B is not None:
                    lines[kd_pos] = f'Kd {kdparm}\n'
                if has_emission:
                    if emission_R is not None or emission_G is not None or emission_B is not None:
                        lines.insert(
                            end_of_material, f'has_Emission {has_emission}\n')
                        lines.insert(
                            end_of_material, f'emission {emissionParm}\n')

        with open(mtl_file_path, 'w') as mtl_file:
            mtl_file.writelines(lines)

    except Exception as e:
        print("Error %s occured during input %s" % (str(e), mesh_path))

    finally:
        imported_objects = bpy.context.selected_objects
        for obj in imported_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        bpy.ops.outliner.orphans_purge()
        for material in bpy.data.materials:
            if material.users == 0:
                bpy.data.materials.remove(material)


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
    parser.add_argument('--output_mesh_folder', type=str,
                        help='output path of generated obj')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]

    # remove all default objects in a collection
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    output_mesh_folder = args.output_mesh_folder
    if not os.path.exists(output_mesh_folder):
        os.mkdir(output_mesh_folder)

    copied_glb_path = os.path.join(output_mesh_folder, mesh_basename + ".glb")
    bpy.ops.import_scene.gltf(filepath=mesh_path)
    time.sleep(0.1)
    bpy.ops.export_scene.gltf(filepath=copied_glb_path)

    glb_convert(copied_glb_path)

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("GLB to obj finish. Start local time is %s, end local time is %s............" % (
        start_time_str, end_time_str))
