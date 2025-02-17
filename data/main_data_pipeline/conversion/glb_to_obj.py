import argparse
import os
import sys
import time

import bpy


def read_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def glb_convert(glb_path: str, glb_folder: str):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    # remove all default objects in a collection
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 导入GLB模型
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # 获取导入的对象
    imported_objects = bpy.context.selected_objects

    # 导出为GLTF
    gltf_path = os.path.join(glb_folder, mesh_basename + ".gltf")
    bpy.ops.export_scene.gltf(filepath=gltf_path, use_selection=True,
                              export_format='GLTF_SEPARATE', export_animations=False)

    # 删除导入的对象
    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # 导入GLTF模型
    bpy.ops.import_scene.gltf(filepath=gltf_path)

    print(gltf_path)

    # 获取导入的对象
    imported_objects = bpy.context.selected_objects

    # 导出为FBX
    obj_path = os.path.join(glb_folder, mesh_basename + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=obj_path, use_selection=True, bake_anim=False)

    # 删除导入的对象
    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # 导入FBX模型
    bpy.ops.import_scene.fbx(filepath=obj_path)

    print(obj_path)

    # 标记是否所有对象的当前材质都满足条件
    all_objects_materials_white = True

    # 遍历所有对象
    for obj in bpy.context.scene.objects:
        # 获取对象的当前材质
        material = obj.active_material
        if material and material.use_nodes:
            for node in material.node_tree.nodes:
                # 检查是否为Principled BSDF节点
                if node.type == 'BSDF_PRINCIPLED':
                    # 检查基础色输入端是否有输入连接
                    if node.inputs['Base Color'].is_linked:
                        all_objects_materials_white = False
                        break
                    else:
                        # 获取基础色颜色的RGB值
                        color = node.inputs['Base Color'].default_value
                        # 检查RGB三通道数值是否相等
                        if color[0] != color[1] or color[0] != color[2]:
                            all_objects_materials_white = False
                            break
            if not all_objects_materials_white:
                break

    # 根据所有对象的当前材质情况设置导出文件名
    output_prefix = "white_" if all_objects_materials_white else ""

    # 获取导入的对象
    imported_objects = bpy.context.selected_objects

    # 导出为OBJ
    obj_path = os.path.join(glb_folder, output_prefix + mesh_basename + ".obj")
    bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True, export_materials=True)

    time.sleep(0.1)

    # 删除导入的对象
    for obj in imported_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # 清理未使用的数据块
    bpy.ops.outliner.orphans_purge()


if __name__ == '__main__':

    t_start = time.time()
    local_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("GLB to obj start. Local time is %s" % (start_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Render data script.')
    parser.add_argument('--mesh_path', type=str,
                        help='path of mesh glb file')
    parser.add_argument('--output_mesh_folder', type=str,
                        help='output path of generated obj')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_mesh_folder = args.output_mesh_folder
    if not os.path.exists(output_mesh_folder):
        os.mkdir(output_mesh_folder)

    glb_convert(mesh_path, output_mesh_folder)

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("GLB to obj finish. Start local time is %s, end local time is %s............" % (
        start_time_str, end_time_str))
