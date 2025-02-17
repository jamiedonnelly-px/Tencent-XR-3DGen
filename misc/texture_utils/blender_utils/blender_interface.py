import logging
import os
import sys

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)

from blender_render_gif.linux_main_renderRotatorMesh import RenderMeshGif


def interface_anything_to_obj(
        input_file,
        output_file,
        blender_path="/usr/blender-3.6.2-linux-x64/blender",
        python_file_name="anything_obj_converter.py"):
    suc_flag = False
    python_file = os.path.join(codedir, python_file_name)
    cmd = f"{blender_path} -b -P {python_file} -- --mesh_path '{input_file}' --output_mesh_path '{output_file}' --copy_texture --mesh_normalization"
    print(f'interface_anything_to_obj cmd = {cmd}')
    os.system(cmd)
    suc_flag = os.path.exists(output_file)
    return suc_flag


def interface_render_gif(input_file, output_file, blender_path="/usr/blender-3.6.2-linux-x64/blender"):
    suc_flag = False
    render_mesh_gif = RenderMeshGif(blender_path=blender_path)
    suc_flag = render_mesh_gif.call_render_gif(input_file, output_file)
    suc_flag = suc_flag and os.path.exists(output_file)
    return suc_flag
