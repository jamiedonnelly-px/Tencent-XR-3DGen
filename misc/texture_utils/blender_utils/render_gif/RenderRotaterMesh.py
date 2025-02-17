import bpy
import bmesh
import math
import mathutils
import sys
import os
import time
import platform
import subprocess
import uuid
from bpy.props import *
import bpy_extras
from mathutils import *
import numpy as np

out_put_dir = str(sys.argv[-1]) #r"E:\\blender\\render\\out\\test"
fbx_path = str(sys.argv[-2])#r"E:\\blender\\render\\mesh3\\2a20b05f-3377-4de1-a65f-8241e918dbaf.glb"


def render_dissolve(out_iamge_path):
    #bpy.context.scene.frame_end = 5
    # bpy.context.scene.render.filepath = out_iamge_path + "\\out" + '.mkv'
    # bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    # bpy.ops.render.render(animation=True, write_still=True)

    # 渲染多帧图片
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 35
    bpy.context.scene.frame_step = 1

    output_path = out_iamge_path
    bpy.context.scene.render.filepath = os.path.join(output_path, "frame_####")
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(animation=True)

    # rendered_images = []
    # for i in range(0, 72, 1):
    #     bpy.context.scene.frame_set(i)

    #     bpy.context.scene.render.filepath = out_iamge_path + "\\" + str(i) + '.png'
    #     bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    #     bpy.ops.render.render(write_still = True)
        # bpy.ops.render.render()
        
        # result = bpy.ops.render.render(write_still=False)
        
        # print(f"Render result: {result}")
                    
        # # 获取渲染结果
        # print('bpy.data.images ', bpy.data.images)
        # print('bpy.data.images keys ', bpy.data.images.keys())  # ['2a20b05f-3377-4de1-a65f-8241e918dbaf', 'autumn_field_puresky_4k.hdr', 'MCWY_2_Bottom___BTM_23', 'Render Result', 'Specular Tint', 'Specular Tint.001', 'textured']
        
        # render_result = bpy.data.images['Render Result']
        # print('render_results ', render_result)

        # # 确保渲染结果已加载到像素缓冲区
        # # render_result.update()

        # # 将渲染结果转换为 NumPy 数组
        # width = render_result.size[0]
        # height = render_result.size[1]
        # num_channels = 4  # RGBA
        # pixels = np.array(render_result.pixels[:])
        # pixels = pixels.reshape((height, width, num_channels))

        # # 将RGBA值转换到0-255范围
        # pixels = (pixels * 255).astype(np.uint8)
        # print('pixels ', pixels)
        # print('filepath', bpy.context.scene.render.filepath)
        # rendered_images.append(pixels)

def RenderMesh(x, y):
    # 设置渲染尺寸
    bpy.context.scene.render.resolution_x = x
    bpy.context.scene.render.resolution_y = y

    # 导入模型
    file_type = os.path.basename(fbx_path).split(".")[1]
    file_name = os.path.basename(fbx_path).split(".")[0]
    print(fbx_path)
    if (file_type.lower() in ['glb', 'gltf']):
        bpy.ops.import_scene.gltf(filepath=fbx_path)
    elif (file_type.lower() in ['fbx']):
        # 下面第一种是导入方式是betterFBX的导入，我移植进来了
        #importFBXFile(fbx_path)
        bpy.ops.import_scene.fbx(filepath=fbx_path)

    bpy.data.objects['Root'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['Root']
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=False)

    tmp_dir = os.path.join(out_put_dir, "tmp")
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    render_dissolve(tmp_dir)



res=768
RenderMesh(res , res)