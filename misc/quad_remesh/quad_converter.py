import os
import time
from math import radians
import json
from quad_remesh_and_bake import remesh_and_bake, load_object, save_object
from quad_to_tri import obj_to_json
import bpy

def clear_workspace():
    # Remove all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Remove all meshes, cameras, lights, etc.
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)

    # Remove all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)


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
 
def quad_remesh_pipeline(input_path, output_path, **kwargs):
    save_fly = False
    time_list = []
    if save_fly:
        ts = time.time()
        fly_dir = os.path.join("/home/tencent/fly", kwargs["job_id"])
        os.makedirs(fly_dir, exist_ok=True)
        time_list.append(("-----cp_in", time.time() -  ts))

    ts = time.time()
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    load_object(input_path)
    time_list.append(("-----load_object", time.time() -  ts))
        
    ts = time.time()
    new_obj = remesh_and_bake(bpy.context.selected_objects[0], **kwargs)
    time_list.append(("-----remesh_and_bake", time.time() -  ts))
    
    # save raw obj
    bpy.ops.object.select_all(action='DESELECT')
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj

    ts = time.time()
    if save_fly:
        dirname, filename = os.path.split(output_path)
        os.makedirs(dirname, exist_ok=True)
        pre, ext = os.path.splitext(filename)        
        fly_out_glb = os.path.join(fly_dir, pre + '.glb')
        fly_out_fbx = os.path.join(fly_dir, pre + '.fbx')
        fly_out_obj = os.path.join(fly_dir, pre + '.obj')
        fly_out_json = os.path.join(fly_dir, pre + '.json')
        
        cos_out_glb = os.path.join(dirname, pre + '.glb')
        cos_out_fbx = os.path.join(dirname, pre + '.fbx')
        cos_out_obj = os.path.join(dirname, pre + '.obj')
        cos_out_json = os.path.join(dirname, pre + '.json')
        
        print('fly_out_glb ', fly_out_glb)
        print('cos_out_glb ', cos_out_glb)
        print('output_path ', output_path)
    
        ## save glb and fbx move y
        if kwargs.get("geom_only", False):
            y_up_mesh_move_y(new_obj, 1.0)
        save_object(fly_out_glb)
        save_object(fly_out_fbx)
        time_list.append(("-----save_glb_fbx", time.time() -  ts))

        ts = time.time()
        if kwargs.get("geom_only", False):
            y_up_mesh_move_y(new_obj, -1.0)
        save_object(fly_out_obj)
        time_list.append(("-----save_obj", time.time() -  ts))
        
        # save tri
        ts = time.time()
        cvt_flag = obj_to_json(fly_out_obj, fly_out_json, move_y=True)
        assert cvt_flag, f"obj_to_json failed, {fly_out_obj}, {fly_out_json}"
        time_list.append(("-----save_json", time.time() -  ts))
        
        ts = time.time()
        os.system(f"cp {fly_dir}/* {dirname}/")
        time_list.append(("-----cp_out", time.time() -  ts))
    else:
        pre, ext = os.path.splitext(output_path)

        ## save glb and fbx move y, need save glb first, i dont know why..
        y_plus = 0
        if kwargs.get("geom_only", False):
            y_plus = y_up_mesh_move_floor(new_obj)
            # y_up_mesh_move_y(new_obj, 1.0)
        save_object(pre + '.glb')
        save_object(pre + '.fbx')
        time_list.append(("-----save_glb_fbx", time.time() -  ts))

        if ext != ".obj":
            out_obj_path = pre + ".obj"
        else:
            out_obj_path = output_path    

        ts = time.time()
        if kwargs.get("geom_only", False):
            y_up_mesh_move_y(new_obj, -y_plus)
            # y_up_mesh_move_y(new_obj, -1.0)
        save_object(out_obj_path)
        time_list.append(("-----save_obj", time.time() -  ts))
        
        
        # save tri
        ts = time.time()
        out_obj_json_path = os.path.splitext(out_obj_path)[0] + ".json"
        cvt_flag = obj_to_json(out_obj_path, out_obj_json_path, y_plus=y_plus)
        assert cvt_flag, f"obj_to_json failed, {out_obj_path}, {out_obj_json_path}"
        time_list.append(("-----save_json", time.time() -  ts))

    print('[INFO] quad_remesh_pipeline time_list: ')
    for data_ in time_list:
        print(data_)
    
    clear_workspace()
    
