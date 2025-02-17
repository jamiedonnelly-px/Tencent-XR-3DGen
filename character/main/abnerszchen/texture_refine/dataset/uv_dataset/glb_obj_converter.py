import bpy
import os
import time

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_obj(file_path):
    bpy.ops.import_scene.obj(filepath=file_path)
    
def import_glb(file_path):
    bpy.ops.import_scene.gltf(filepath=file_path)

def export_glb(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=file_path, export_format='GLB', use_selection=True)

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



# ref: dataset/control_pre/obj_to_glb.py
def convert_obj_to_glb(input_obj_path, output_glb_path):
    try:
        if not os.path.exists(input_obj_path):
            print(f"can not find input_obj_path {input_obj_path}")
            return False            
        clear_scene()
        import_obj(input_obj_path)
            
        imported_objs = bpy.context.selected_objects
        for obj in imported_objs:
            obj.select_set(True)        
        export_glb(output_glb_path)
        return True
    except Exception as e:
        print("convert_obj_to_glb failed", e)
        return False

# for grpc with blender env
def blender_worker(request_queue, response_queue):
    while True:
        request = request_queue.get()

        if request == "shutdown":
            break 

        if request["type"] == "convert_glb_to_obj":
            input_file = request["input_file"]
            output_file = request["output_file"]
            result = convert_glb_to_obj(input_file, output_file)
            print('[blender_worker] convert_glb_to_obj')
            
        elif request["type"] == "convert_obj_to_glb":
            input_file = request["input_file"]
            output_file = request["output_file"]
            result = convert_obj_to_glb(input_file, output_file)
            print('[blender_worker] convert_obj_to_glb')

        response_queue.put(result)
        