import bpy
import os
import time
from math import radians
import math

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_obj(file_path):
    bpy.ops.import_scene.obj(filepath=file_path)

def load_mesh(mesh_path: str):
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    mesh_extension_lower = mesh_extension.lower()
    if mesh_extension_lower == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path, use_anim=True)
    elif mesh_extension_lower == ".obj":
        version_info = bpy.app.version
        if version_info[0] > 2:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        else:
            bpy.ops.import_scene.obj(filepath=mesh_path)


def import_glb(file_path):
    bpy.ops.import_scene.gltf(filepath=file_path)

def export_glb(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=file_path, export_format='GLB', use_selection=True)

def export_fbx(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # bpy.ops.export_scene.fbx(filepath=file_path, use_selection=True)
    bpy.ops.export_scene.fbx(filepath=file_path, use_selection=True, path_mode='COPY', embed_textures=True)

def rotate_mesh_x(mesh_object, angle_degrees):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.transform.rotate(value=radians(angle_degrees), orient_axis='X')

def rotate_mesh_z(mesh_object, angle_degrees):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.transform.rotate(value=radians(angle_degrees), orient_axis='Z')

def translate_mesh_axis(mesh_object, axis, distance):
    """translate mesh along the axis

    Args:
        mesh_object: _description_
        axis: can be x, y, z
        distance: meter
    """
    bpy.ops.object.select_all(action='DESELECT')
    mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = mesh_object
    if axis == "x":
        mesh_object.location.x += distance
    elif axis == "y":
        mesh_object.location.y += distance
    elif axis == "z":
        mesh_object.location.z += distance

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print('transform_apply ')

def calcu_axis(obj):
    max_vals = [-float('inf')] * 3
    min_vals = [float('inf')] * 3
    suc_flag = False
    try:
        ts = time.time()
        mesh = obj.data
        for vertex in mesh.vertices:
            co = vertex.co

            for i in range(3):
                max_vals[i] = max(max_vals[i], co[i])
                min_vals[i] = min(min_vals[i], co[i])
        suc_flag = True
    except Exception as e:
        print(f"calcu_axis failed , e = {e}")

    print("物体在各个轴上的最大值：", max_vals)
    print("物体在各个轴上的最小值：", min_vals)
    print("use time ", time.time() - ts)
    return suc_flag, min_vals, max_vals

def y_up_mesh_move_y(mesh_object, distance):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.transform.rotate(value=radians(90), orient_axis='X')
    mesh_object.location.y += distance

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.transform.rotate(value=radians(-90), orient_axis='X')
    print('y_up_mesh_move_y ')

def y_up_mesh_move_floor(mesh_object, y_offset=0.02):
    flag, min_vals, max_vals = calcu_axis(mesh_object)
    y_plus = 0
    if flag:
        y_plus = -min_vals[1] + y_offset
        y_up_mesh_move_y(mesh_object, y_plus)
        calcu_axis(mesh_object)
    else:
        print(f"calcu_axis failed , skip move")
    return y_plus

def mesh_shade_smooth(object, angle=30):
    version_info = bpy.app.version
    print("version_info", version_info)
    if version_info[0] >= 4 and version_info[1] >= 1:
        node = object.modifiers.new("Smooth by Angle", "NODES")
        result = bpy.ops.object.modifier_add_node_group(asset_library_type='ESSENTIALS', asset_library_identifier="",
                                                        relative_asset_identifier="geometry_nodes\\smooth_by_angle.blend\\NodeTree\\Smooth by Angle")
        if 'CANCELLED' in result:
            return

        modifier = object.modifiers[-1]
        modifier["Socket_1"] = True
        modifier["Input_1"] = math.radians(angle)
        object.update_tag()
    else:
        object.data.use_auto_smooth = True
        object.data.auto_smooth_angle = math.radians(angle)
    for f in object.data.polygons:
        f.use_smooth = True

# ref: dataset/control_pre/glb_to_obj.py
def convert_glb_to_obj(input_glb_path, output_obj_path):
    try:
        if not os.path.exists(input_glb_path):
            print(f"can not find input_glb_path {input_glb_path}")
            return False
        mesh_folder = os.path.split(input_glb_path)[0]
        mesh_filename = os.path.split(input_glb_path)[1]
        mesh_basename = os.path.splitext(mesh_filename)[0]
        mesh_extension = os.path.splitext(mesh_filename)[1]
        out_folder = os.path.dirname(output_obj_path)
        os.makedirs(out_folder, exist_ok=True)

        ## begin
        clear_scene()
        # 导入GLB模型
        bpy.ops.import_scene.gltf(filepath=input_glb_path)

        # 获取导入的对象
        imported_objects = bpy.context.selected_objects

        # 导出为GLTF
        gltf_path = os.path.join(out_folder, mesh_basename + ".gltf")
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
        obj_path = os.path.join(out_folder, mesh_basename + ".fbx")
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
        obj_path = output_obj_path
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        # obj_path = os.path.join(out_folder, output_prefix+mesh_basename + ".obj")
        bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True, export_materials=True)

        time.sleep(0.1)

        # 删除导入的对象
        for obj in imported_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # 清理未使用的数据块
        bpy.ops.outliner.orphans_purge()

        # done
        return True
    except Exception as e:
        print("convert_glb_to_obj failed", e)
        return False


def toggle_alpha_blend_mode(object, blend_method='OPAQUE'):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                slot.material.blend_method = blend_method
                print('toggle_alpha_blend_mode done')


# ref: dataset/control_pre/obj_to_glb.py
def convert_obj_to_glb(input_obj_path, output_glb_path, mesh_post_processing=None):
    try:
        if not os.path.exists(input_obj_path):
            print(f"can not find input_obj_path {input_obj_path}")
            return False
        clear_scene()
        load_mesh(input_obj_path)

        imported_objs = bpy.context.selected_objects

        process_operations = {
            "z-up-to-y-up": (lambda obj: rotate_mesh_x(obj, 90), "rot x 90"),
            "y-up-to-z-up": (lambda obj: rotate_mesh_x(obj, -90), "rot x -90"),
            "lrm-to-opengl":
            (lambda obj: [rotate_mesh_x(obj, 90),
                          rotate_mesh_z(obj, 90),
                          translate_mesh_axis(obj, "z", 1.0)],
             "lrm-to-opengl, rot to y up and translate to y in [0, 2]"),
            "move-up-y": (lambda obj: y_up_mesh_move_floor(obj), "move-up-y, translate to y in [0, 2]"),
            "move-up-y-smooth":
            (lambda obj: [y_up_mesh_move_floor(obj), mesh_shade_smooth(obj)],
             "move-up-y-smooth, translate to y in [0, 2] and smooth"),
            "after-fast-baking": (lambda obj: None, ""),
            "export-fbx": (lambda obj: None, ""),
        }

        if mesh_post_processing is not None and mesh_post_processing:
            operation, message = process_operations.get(mesh_post_processing, (lambda obj: None, ""))

            print(f"{message} with mesh_post_processing={mesh_post_processing}")
            for obj in imported_objs:
                operation(obj)

        for obj in imported_objs:
            toggle_alpha_blend_mode(obj)

        for obj in imported_objs:
            obj.select_set(True)

        os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
        export_glb(output_glb_path)

        if mesh_post_processing in ["lrm-to-opengl", "after-fast-baking", "export-fbx", "move-up-y", "move-up-y-smooth"]:
            output_fbx_path = output_glb_path.replace(".glb", ".fbx")
            export_fbx(output_fbx_path)
        return True
    except Exception as e:
        print("convert_obj_to_glb failed", e)
        return False

def blender_worker(request_queue, response_queue):
    while True:
        request = request_queue.get()

        if request == "shutdown":
            break

        if request["type"] == "convert_glb_to_obj":
            input_file = request["input_file"]
            output_file = request["output_file"]
            result = convert_glb_to_obj(input_file, output_file)
            print(f'[blender_worker] convert_glb_to_obj done {input_file} -> {output_file}')

        elif request["type"] == "convert_obj_to_glb":
            input_file = request["input_file"]
            output_file = request["output_file"]
            mesh_post_processing = request.get("mesh_post_processing", None)
            result = convert_obj_to_glb(input_file, output_file, mesh_post_processing)
            print(f'[blender_worker] convert_obj_to_glb done {input_file} -> {output_file} with {mesh_post_processing}')

        response_queue.put(result)
