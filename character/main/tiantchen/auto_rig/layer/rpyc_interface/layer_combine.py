import os, re, sys, bpy, json, bmesh, mathutils
import numpy as np
from math import radians
from sklearn.neighbors import NearestNeighbors
from mathutils.bvhtree import BVHTree

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
sys.path.insert(0, root_path)
from utils.hair_color_change import change_hair, material_with_Kd
from utils.utils_blender import utils, base_func, fix_mesh_transparency, names
from utils.rename_openxr import rename_openxr

class MyClass:
    def __init__(self, input_path, is_local=False):
        self.input_path = input_path
        self.is_local = is_local
        local_root = "/Users/chentian/Downloads/code/data/combine"

        # body 
        body_file = open(os.path.join(self.input_path,"smplx-path.txt"), "r").read().split("\n")
        self.body_path = body_file[0].replace("/aigc_cfs/rabbityli", "/aigc_cfs_gdp/Asset")
        if(self.is_local):
            self.body_path = self.body_path.replace('/aigc_cfs/rabbityli', local_root)
        self.body_obj_path = os.path.join(self.body_path, "naked")
        self.body_type = self.body_obj_path.split('/')[-2].split('_')[0]

        # part
        self.part_paths=[]
        part_json = open(os.path.join(self.input_path,"warp_lst.json"), "r")
        part_files = json.load(part_json)
        for part_file in part_files:
            class_name = part_files[part_file]
            if part_file != '':
                if self.is_local:
                    part_file = os.path.join(local_root, "data", part_file.split('/')[-2], part_file.split('/')[-1])
                if os.path.exists(part_file):
                    if self.body_type == "yuanmeng" and class_name == "trousers":
                        print("test")
                    else:
                        self.part_paths.append([part_file, class_name])    

        # body mask
        body_mask_json = os.path.join(self.body_path, "auto_rig/mask_faces.json")
        with open(body_mask_json, 'r') as file:
            self.mask_data = json.load(file)

        # loop
        self.body_loop = []
        for loop in self.mask_data["body"]["loop"]:
            for lp in self.mask_data["body"]["loop"][loop]:
                self.body_loop.append(lp)

        # neck
        self.neck = []
        for lp in self.mask_data["body"]["loop"]["neck"]:
            self.neck += lp
            
        # name
        self.part_names = []

        if is_local:
            smplx_seg_file = "/Users/chentian/Downloads/code/data/smplx_vert_segmentation_pro.json"
        else:
            smplx_seg_file = os.path.join(root_path, "utils/smplx_vert_segmentation_pro.json")
        smplx_data = json.load(open(smplx_seg_file))
        # smplx 头部idx
        self.smplx_head_idx = smplx_data["pro"]["head"]
        self.smplx_left_verts_idx = smplx_data["pro"]["left"]
        self.smplx_right_verts_idx = smplx_data["pro"]["right"]

    def is_highheel(self):
        total_data_path = "/aigc_cfs_gdp/list/active_list2/layered_data/20240711/20240711_ruku_ok_gdp.json"
        with open(total_data_path,'r') as files:
            total_data = json.load(files)

        is_high_heel = False
        with open(os.path.join(self.input_path, "object_lst.txt"), 'r') as file:
            content = file.read()
            obj_info_data = json.loads(content)
            for part in obj_info_data["path"]:
                if obj_info_data["path"][part]["cat"] == "shoe":
                    asset_key = obj_info_data["path"][part]["asset_key"]
                    for key_type in total_data["data"]:
                        for key in total_data["data"][key_type]:
                            if key == asset_key:
                                print(asset_key)
                                if "HighHeel" in total_data["data"][key_type][key]:
                                    is_high_heel = total_data["data"][key_type][key]["HighHeel"]
        print("===========", is_high_heel)
        return is_high_heel


    def change_hair_color(self):
        hair_color_prompt = ''
        with open(os.path.join(self.input_path, "object_lst.txt"), 'r') as file:
            content = file.read()
            obj_info_data = json.loads(content)
            if "hair_color" in obj_info_data:
                hair_color_prompt = obj_info_data["hair_color"]
        if not hair_color_prompt:
            return
        if hair_color_prompt == "---":
            return
        # change
        for part_pair in self.part_paths:
            if part_pair[1] == "hair" and hair_color_prompt:
                # 判断是否可改变颜色
                for key in obj_info_data["path"]:
                    if "hair" == obj_info_data["path"][key]["cat"]:                           
                        mtl_file = os.path.join(part_pair[0], "material.mtl")
                        
                        with open(mtl_file, 'r') as file:
                            content = file.read()
                        file.close()

                        kd_pattern = re.compile(r'Kd (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)')
                        match = kd_pattern.search(content)

                        new_kd_color = change_hair(hair_color_prompt)
                        new_kd_color_str = " ".join([str(value) for value in new_kd_color])
                        content = kd_pattern.sub(f'Kd {new_kd_color_str}', content)
                        
                        with open(mtl_file, 'w') as file:
                            file.write(content)
                        file.close()

                        with open(os.path.join(self.input_path,"change_color.txt"), 'w') as file_1:
                            file_1.write("change hair color \n")
                            file_1.write(mtl_file + "\n")
                            file_1.write("init: "+  match.group() +"\n")
                            file_1.write("change: " + new_kd_color_str+"\n")
                    
    def import_part_one(self, file_name):
        # 记录当前所有mesh
        scene_objs = []
        for mesh in bpy.context.scene.objects:
             if mesh.type == 'MESH':
                 scene_objs.append(mesh.name)

        # import part   
        if file_name[1] == "shoe":
            utils.import_file(os.path.join(file_name[0], file_name[0].split('/')[-1]+'_left.obj'))
            utils.import_file(os.path.join(file_name[0], file_name[0].split('/')[-1]+'_right.obj'))
            # 重命名 part
            for mesh in bpy.context.scene.objects:
                if mesh.type == 'MESH'and mesh.name not in scene_objs:
                    if "left" in mesh.name:
                        mesh.name = "shoe_left"
                    if "right" in mesh.name:
                        mesh.name = "shoe_right"
            # 记录所有part name
            self.part_names.append("shoe_left")
            
            # rotate  (修改)
            rotation_axis = 'X'
            rotation_angle = -90 
            obj = bpy.data.objects["shoe_left"]
            mesh = obj.data
            rotation_matrix = mathutils.Matrix.Rotation(radians(rotation_angle), 4, rotation_axis)
            mesh.transform(rotation_matrix)
            mesh.update()
            
            # 记录所有part name
            self.part_names.append("shoe_right")
            # rotate  (修改)
            rotation_axis = 'X'
            rotation_angle = -90 
            obj = bpy.data.objects["shoe_right"]
            mesh = obj.data
            rotation_matrix = mathutils.Matrix.Rotation(radians(rotation_angle), 4, rotation_axis)
            mesh.transform(rotation_matrix)
            mesh.update()
        else:
            utils.import_file(os.path.join(file_name[0], file_name[0].split('/')[-1]+'.obj'))
            # 重命名 part
            part_objs = []
            for mesh in bpy.context.scene.objects:
                if mesh.type == 'MESH' and mesh.name not in scene_objs:
                    part_objs.append(bpy.context.scene.objects[mesh.name])
            if all(obj.type == 'MESH' for obj in part_objs):
                bpy.ops.object.select_all(action='DESELECT')
                for obj in part_objs:
                    obj.select_set(True)
                bpy.context.view_layer.objects.active = part_objs[0]
                bpy.ops.object.join()
            for mesh in bpy.context.scene.objects:
                if mesh.type == 'MESH' and mesh.name not in scene_objs:
                    mesh.name = file_name[1]
            # 记录所有part name
            self.part_names.append(file_name[1])

            # rotate  (修改)
            rotation_axis = 'X'
            rotation_angle = -90 
            obj = bpy.data.objects[file_name[1]]
            mesh = obj.data
            rotation_matrix = mathutils.Matrix.Rotation(radians(rotation_angle), 4, rotation_axis)
            mesh.transform(rotation_matrix)
            mesh.update()
        
    def import_part(self):
        flag = False
        for part_file in self.part_paths:
            if part_file[1] == "outfit":
                flag = True
                self.import_part_one(part_file)
            if part_file[1] in ["hair", "shoe"]:
                self.import_part_one(part_file)
        if not flag:
            for part_file in self.part_paths:
                if part_file[1] in ["top", "trousers"]:
                    self.import_part_one(part_file)
        
    def export(self): 
        # 删除smplx 
        for obj in bpy.context.scene.objects:
            if obj.name in [names["smplx_armature"], names["smplx_mesh"]]:
                bpy.ops.object.select_all(action='DESELECT') 
                object_to_delete = bpy.data.objects.get(obj.name)
                object_to_delete.select_set(True)
                bpy.context.view_layer.objects.active = object_to_delete
                bpy.ops.object.delete()  

        out_path = os.path.join(self.input_path, "mesh")
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            
        # 输出人体骨骼面片数
        model_info_json = os.path.join(out_path, "model_info.json")
        data = {}
        head_obj = bpy.data.objects.get(names["head"])
        body_obj = bpy.data.objects.get(names["body"])
        data["head_face_count"] = len(head_obj.data.polygons)
        data["body_face_count"] = len(body_obj.data.polygons)
        with open(model_info_json, 'w') as f:
            json.dump(data, f) 

        # 修复glb透明度
        fix_mesh_transparency.run(out_path)

         # 按照openxr的命名方式输出
        rename_openxr()  


        if self.is_highheel():
            # 使用骨骼的名字来获取骨骼对象
            armature = bpy.data.objects['Root'] # 请确保这里的'Armature'是你的骨骼对象的名字
            armature.pose.bones["RightFoot"].rotation_quaternion = [0.966, 0.259, 0, 0] 
            armature.pose.bones["LeftFoot"].rotation_quaternion = [0.966, 0.259, 0, 0] 
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)

                    for modifier in obj.modifiers:
                        bpy.ops.object.modifier_apply({"object": obj}, modifier=modifier.name)
                    # bind mesh
                    bpy.ops.object.select_all(action='SELECT')
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                    bpy.context.view_layer.update()
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = armature
                    obj.parent = armature
                    obj.parent_type = 'ARMATURE'
                    armature_modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
                    armature_modifier.object = armature

        # amature init
        # bpy.ops.object.select_all(action='DESELECT')
        # bpy.context.view_layer.objects.active = armature
        # armature.select_set(True)
        # bpy.ops.object.mode_set(mode='POSE')
        # bpy.ops.pose.select_all(action='SELECT')
        # bpy.ops.pose.armature_apply(selected=False)
        # bpy.context.view_layer.update()
        # bpy.ops.object.mode_set(mode='OBJECT')

        # # 导出带骨骼数据的fbx
        # export_path = utils.export_fbx(os.path.join(out_path, "mesh.fbx"))
            
        # if not self.is_local:
        #     # 删除骨骼数据后导出为glb
        #     for obj in bpy.context.scene.objects:
        #          if obj.name in [names["armature"]]:
        #              bpy.ops.object.select_all(action='DESELECT') 
        #              object_to_delete = bpy.data.objects.get(obj.name)
        #              object_to_delete.select_set(True)
        #              bpy.context.view_layer.objects.active = object_to_delete
        #              bpy.ops.object.delete()  

        export_path = os.path.join(out_path, "mesh.glb")
        bpy.ops.export_scene.gltf(filepath=export_path, export_format= "GLB", use_selection=False)
        # gltf_path = os.path.join(out_path, "mesh.gltf")
        # bpy.ops.export_scene.gltf(filepath=gltf_path, export_format='GLTF_SEPARATE', use_selection=False)

        
            
    def split_faces_shoe(self, th = 0.005):        
        body_object = bpy.data.objects[names["body"]]
        part_object = bpy.data.objects["shoe"]
        bm_part = bmesh.new() 
        bm_part.from_mesh(part_object.data)
        bvh_part = BVHTree.FromBMesh(bm_part)
        
        bm_body = bmesh.new()
        bm_body.from_mesh(body_object.data)            
        bm_body.faces.ensure_lookup_table()
        bm_body.edges.ensure_lookup_table()
        bm_body.verts.ensure_lookup_table()
        bm_body.normal_update()
        
        # 检查射线是否与mesh part相交
        for i in range(len(bm_body.verts)):
            vert = bm_body.verts[i]    
      
            vertices = []
            ray_direction = vert.normal
            
            location, normal, index, distance = bvh_part.ray_cast(vert.co, -ray_direction)
            
            if location:
                if distance < th:
                    bm_body.verts[i].co = vert.co - distance*ray_direction*2

        bm_body.to_mesh(body_object.data)
        bm_body.free()
        bm_part.free()  
        bpy.context.view_layer.update()
        
    def split_faces_head(self):
        body_name = names["head"]
        body_object = bpy.data.objects[body_name]
        bm_body = bmesh.new()
        bm_body.from_mesh(body_object.data)            
        bm_body.faces.ensure_lookup_table()
        bm_body.edges.ensure_lookup_table()
        bm_body.verts.ensure_lookup_table()
            
        remove_lists = []
        th = 0.2
            
        part_object = bpy.data.objects["hair"]
        bm_part = bmesh.new() 
        bm_part.from_mesh(part_object.data)
        bvh_part = BVHTree.FromBMesh(bm_part)
            
        # 检查射线是否与mesh part相交
        for i in range(len(bm_body.faces)):
            if i in self.mask_data["head"]["undelete"]:
                continue
  
            face = bm_body.faces[i]                    
            vertices = []
            for loop in face.loops:
                vertices.append(loop.vert)
                
            ray_direction = base_func.calculate_polygon_normal(vertices)
            flag = True
            for viewpoint in vertices:
                location, normal, index, distance = bvh_part.ray_cast(viewpoint.co, ray_direction)
                location_1, normal_1, index_1, distance_1 = bvh_part.ray_cast(viewpoint.co, -ray_direction)

                if distance is None and distance_1 is None:
                    flag = False
                elif distance is None:
                    if distance_1 > th:
                        flag = False
                elif distance_1 is None:
                    if distance > th:
                        flag = False
                else:
                    if distance > th and distance_1 > th:
                        flag = False
            if flag:
                remove_lists.append(face.index) 
        bm_part.free()   
            
        remove_lists_with_loop = []
        for loop in self.body_loop:
            if all(item in remove_lists for item in loop):
                remove_lists_with_loop += loop
        for idx in remove_lists:
            if not any(idx in row for row in self.body_loop):
                remove_lists_with_loop.append(idx)
        
        remove_lists = remove_lists_with_loop + self.mask_data["head"]["delete"]
            
        remove_faces = []
        remove_sets = set(remove_lists)
        for idx in remove_sets:
            remove_faces.append(bm_body.faces[idx])
        bmesh.ops.delete(bm_body, geom=remove_faces, context='FACES')

        # 更新视图
        bm_body.to_mesh(body_object.data)
        bm_body.free()
        bpy.context.view_layer.update()
        
    def split_faces_part(self):        
        parts = []
        for part_name in self.part_names:
            if part_name not in ["hair"]:
                parts.append(part_name)
        
        body_name = names["body"]
        body_object = bpy.data.objects[body_name]
        bm_body = bmesh.new()
        bm_body.from_mesh(body_object.data)            
        bm_body.faces.ensure_lookup_table()
        bm_body.edges.ensure_lookup_table()
        bm_body.verts.ensure_lookup_table()
        
        remove_lists = []
        neck_th = 0.05
        body_th = 0.3
        th = body_th
        for part_name in parts:
            part_object = bpy.data.objects[part_name]
            bm_part = bmesh.new() 
            bm_part.from_mesh(part_object.data)
            bvh_part = BVHTree.FromBMesh(bm_part)
        
            # 检查射线是否与mesh part相交
            for i in range(len(bm_body.faces)):
                if i in self.mask_data["body"]["undelete"]: #手
                    continue
                if i in self.neck: #脖子
                    th = neck_th
                if i in self.mask_data["body"]["foot"]: #脚
                    continue
                    
                face = bm_body.faces[i]                    
                vertices = []
                for loop in face.loops:
                    vertices.append(loop.vert)
                    
                ray_direction = base_func.calculate_polygon_normal(vertices)
                flag = True
                
                for viewpoint in vertices:
                    location, normal, index, distance = bvh_part.ray_cast(viewpoint.co, ray_direction)
                    location_1, normal_1, index_1, distance_1 = bvh_part.ray_cast(viewpoint.co, -ray_direction)

                    if distance is None and distance_1 is None:
                        flag = False
                    elif distance is None:
                        if distance_1 > th:
                            flag = False
                    elif distance_1 is None:
                        if distance > th:
                            flag = False
                    else:
                        if distance > th and distance_1 > th:
                            flag = False
                if flag:
                    remove_lists.append(face.index)
            bm_part.free()   
                
        remove_lists_with_loop = []
        for loop in self.body_loop:
            if all(item in remove_lists for item in loop):
                remove_lists_with_loop += loop
        for idx in remove_lists:
            if not any(idx in row for row in self.body_loop):
                remove_lists_with_loop.append(idx)
        
        remove_lists = remove_lists_with_loop + self.mask_data["body"]["delete"]
            
        remove_faces = []
        remove_sets = set(remove_lists)
        for idx in remove_sets:
            remove_faces.append(bm_body.faces[idx])
        bmesh.ops.delete(bm_body, geom=remove_faces, context='FACES')

        # 更新视图
        bm_body.to_mesh(body_object.data)
        bm_body.free()
        bpy.context.view_layer.update()
            
    def skinning_weight_knn_seg_lr(self, armature_object, mesh_object, mesh_smplx_object, mesh_name):
        mesh_object.vertex_groups.clear()
        bones = armature_object.data.bones
        # create group
        for bone in bones:
           vertex_group = mesh_object.vertex_groups.new(name=bone.name)
           
        mesh_vertices_list = [v.co for v in mesh_object.data.vertices]
        mesh_vertices_array = np.array(mesh_vertices_list)
               
        smplx_vertices_list = [v.co for v in mesh_smplx_object.data.vertices]
        smplx_vertices_array = np.array(smplx_vertices_list)

        if mesh_name in ["hair"]:
            smplx_vertices_array_seg = smplx_vertices_array[self.smplx_head_idx]
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(smplx_vertices_array_seg)
            nearest_indices = knn.kneighbors(mesh_vertices_array, return_distance=False)    
            for i in range(len(nearest_indices)):
                min_index = nearest_indices[i][0] 
                idx = self.smplx_head_idx[min_index]
                vertex_in_groups = []
                for vertex_group in mesh_smplx_object.vertex_groups:
                    try:
                        weight = vertex_group.weight(idx)
                        vertex_in_groups.append((vertex_group.name, weight))
                    except:
                        pass
                for vertex_in_group in vertex_in_groups:
                    vertex_group = mesh_object.vertex_groups.get(vertex_in_group[0])
                    vertex_group.add([i], vertex_in_group[1], 'REPLACE')
        elif "left" in mesh_name:
            smpl_indices_idx = self.smplx_left_verts_idx
            smplx_vertices_array_lr = smplx_vertices_array[smpl_indices_idx]
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(smplx_vertices_array_lr)
            nearest_indices = knn.kneighbors(mesh_vertices_array, return_distance=False)    
            for i in range(len(nearest_indices)):
                min_index = nearest_indices[i][0]
                min_index_init = smpl_indices_idx[min_index]
                idx = i
                vertex_in_groups = []
                for vertex_group in mesh_smplx_object.vertex_groups:
                    try:
                        weight = vertex_group.weight(min_index_init)
                        vertex_in_groups.append((vertex_group.name, weight))
                    except:
                        pass
                for vertex_in_group in vertex_in_groups:
                    if vertex_in_group[1] > 0.01:
                        vertex_group = mesh_object.vertex_groups.get(vertex_in_group[0])
                        vertex_group.add([idx], vertex_in_group[1], 'REPLACE') 
        elif "right" in mesh_name:
            smpl_indices_idx = self.smplx_right_verts_idx
            smplx_vertices_array_lr = smplx_vertices_array[smpl_indices_idx]
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(smplx_vertices_array_lr)
            nearest_indices = knn.kneighbors(mesh_vertices_array, return_distance=False)    
            for i in range(len(nearest_indices)):
                min_index = nearest_indices[i][0]
                min_index_init = smpl_indices_idx[min_index]
                idx = i
                vertex_in_groups = []
                for vertex_group in mesh_smplx_object.vertex_groups:
                    try:
                        weight = vertex_group.weight(min_index_init)
                        vertex_in_groups.append((vertex_group.name, weight))
                    except:
                        pass
                for vertex_in_group in vertex_in_groups:
                    if vertex_in_group[1] > 0.01:
                        vertex_group = mesh_object.vertex_groups.get(vertex_in_group[0])
                        vertex_group.add([idx], vertex_in_group[1], 'REPLACE') 
        else:
            lr = [True, False]
            for is_left in lr:
                if is_left:
                    smpl_indices_idx = self.smplx_left_verts_idx
                    mesh_lr_indices = np.nonzero(mesh_vertices_array[:, 0] > 0)[0]
                else:
                    smpl_indices_idx = self.smplx_right_verts_idx
                    mesh_lr_indices =  [i for i in range(len(mesh_vertices_array)) if i not in np.nonzero(mesh_vertices_array[:, 0] > 0)[0]]
                mesh_vertices_array_lr = mesh_vertices_array[mesh_lr_indices]
                smplx_vertices_array_lr = smplx_vertices_array[smpl_indices_idx]
                knn = NearestNeighbors(n_neighbors=1)
                knn.fit(smplx_vertices_array_lr)
                nearest_indices = knn.kneighbors(mesh_vertices_array_lr, return_distance=False)    
                for i in range(len(nearest_indices)):
                    min_index = nearest_indices[i][0]
                    min_index_init = smpl_indices_idx[min_index]
                    idx = int(mesh_lr_indices[i])
                    vertex_in_groups = []
                    for vertex_group in mesh_smplx_object.vertex_groups:
                        try:
                            weight = vertex_group.weight(min_index_init)
                            vertex_in_groups.append((vertex_group.name, weight))
                        except:
                            pass
                    for vertex_in_group in vertex_in_groups:
                        if vertex_in_group[1] > 0.01:
                            vertex_group = mesh_object.vertex_groups.get(vertex_in_group[0])
                            vertex_group.add([idx], vertex_in_group[1], 'REPLACE')

    def skinning_part(self):
        armature = bpy.data.objects[names["armature"]]
        smplx_mesh = bpy.data.objects[names["smplx_mesh"]]

        for mesh in bpy.context.scene.objects:
            if mesh.type == 'MESH':
                if "SMPLX" not in mesh.name:
                    if names["head"] in mesh.name:
                        mesh.name = names["head"]
                    elif names["body"] in mesh.name:
                        mesh.name = names["body"]
                    else:
                        for modifier in mesh.modifiers:
                            bpy.ops.object.modifier_remove({"object":mesh}, modifier=modifier.name)
                        for vertex_group in mesh.vertex_groups:
                            mesh.vertex_groups.remove(vertex_group)
                        base_func.bind_mesh_armature(mesh, armature)
                        self.skinning_weight_knn_seg_lr(armature, mesh, smplx_mesh, mesh.name)
                        # prune_mesh_all(mesh)
            elif mesh.type == 'ARMATURE':
                armature = mesh.data
                bpy.context.view_layer.objects.active = mesh
                bpy.ops.object.mode_set(mode='EDIT')

                # Loop through the bones
                for bone in armature.edit_bones:
                    if "end_end" in bone.name:
                        armature.edit_bones.remove(bone)
                bpy.ops.object.mode_set(mode='OBJECT')


    def clear_scene(self):
        # Ensure we are in Object mode
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()


    # Function to find the armature in the scene
    def find_obj(self, type="ARMATURE", inverse=False):
        objs = []
        for obj in bpy.context.scene.objects:
            if not inverse and obj.type == type:
                objs.append(obj)
            elif inverse and obj.type != type:
                objs.append(obj)
        return objs

    def load_armature(self, body_arm_path):
        
        try:
            # Set up the import settings
            bpy.ops.import_scene.fbx(
                filepath=body_arm_path,
                force_connect_children=True,
                ignore_leaf_bones=True
            )
        except Exception as e:
            print(f'error loading fbx {e}')
        
        bpy.ops.object.select_all(action='DESELECT')
        # load template armature
        armature =self.find_obj("ARMATURE")[0]
        bpy.context.object.show_in_front = True
        bpy.context.object.data.show_axes = True
        # delete other objects
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_all(action='SELECT')
        armature.select_set(False)
        bpy.ops.object.delete()
        
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        
        #delete the root bone
        bpy.ops.armature.select_all(action='DESELECT')
        first_bone = armature.data.edit_bones[0]
        first_bone.select = True
        bpy.ops.armature.delete()
        
        #delete the end bone
        for bone in armature.data.edit_bones:
            bpy.ops.armature.select_all(action='DESELECT')
            if '_end' in bone.name:
                bone.select = True
                bpy.ops.armature.delete()
        
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return armature

    def auto_weight(self):

        """Step 1, per parts auto_rigging"""
        clear_scene()
        smplx_path = self.input_path + '/smplx-path.txt'
        with open(smplx_path, 'r') as file:
            smplx_path = file.readline().strip()
        body_arm_path = smplx_path + '/auto_rig/armature_mesh.fbx'
        print("loading body armature from:", body_arm_path)
        
        # load cloth object
        with open(f'{self.input_path}/warp_lst.json', 'r') as f:
            cloth_data = json.load(f)
        auto_rigging = {}
        for k, v in cloth_data.items():
            self.clear_scene()
            armature = self.load_armature(body_arm_path)
            if v == 'hair':
                armature.select_set(True)
                bpy.context.view_layer.objects.active = armature
                bpy.ops.object.mode_set(mode='EDIT')
                #delete the body bone for hair
                for bone in armature.data.edit_bones:
                    bpy.ops.armature.select_all(action='DESELECT')
                    if bone.name != 'head' and bone.name != 'jaw':
                        bone.select = True
                        bpy.ops.armature.delete()
            bpy.ops.object.mode_set(mode='OBJECT')
                    
            part_dir = f"{self.input_path}/{k.split('/')[-1]}/"
            obj_files = glob.glob(os.path.join(part_dir, "*.obj"))
            for obj_file in obj_files:
                print(f"load {obj_file}")
                bpy.ops.wm.obj_import(filepath=obj_file)
            
            mesh_objs =self.find_obj("MESH")
            for obj in mesh_objs:
                obj.rotation_mode = "XYZ"
                obj.rotation_euler[0] = 0
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                # combine duplicated vertices
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.remove_doubles(threshold=0.01)
                bpy.ops.object.mode_set(mode='OBJECT')
                
            for obj in mesh_objs:
                obj.select_set(True)
                
            # Bind mesh to armature (skinning)
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
                
            joint_names = []
            for bone in armature.data.bones:
                joint_names.append(bone.name)
            joint_names = np.array(joint_names)
            nb = len(joint_names)
            objs = self.find_obj('MESH')
            
            for obj in objs:
                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_mesh = obj.evaluated_get(depsgraph)
                mesh = eval_mesh.data
                # Access vertex groups (bones) attached to this mesh
                vertex_groups = obj.vertex_groups
                n_verts = len(mesh.vertices)
                w = np.zeros([n_verts, nb])
                vs = np.zeros([n_verts, 3]) 
                for vgroup in vertex_groups:
                    ji = np.where(vgroup.name==joint_names)[0].item()
                    for vi, v in enumerate(mesh.vertices):
                        # Iterate through each vertex group and get the weight for this vertex
                        vs[vi] = np.array(list(obj.matrix_world @ v.co))
                        for gp in v.groups:
                            if gp.group == vgroup.index:
                                w[vi,ji] = gp.weight
                            
                # Clean up: free the evaluated mesh to avoid memory leaks
                eval_mesh.to_mesh_clear()    
                
                faces = []
                for triangle in obj.data.polygons:
                    faces.append([v for v in triangle.vertices])
                faces = np.array(faces)
                
                auto_rigging[f'weights_{obj.name}'] = {
                                    "verts": vs, 
                                    "faces": faces, 
                                    "weights": w,
                                    "joint_names":joint_names
                                    }

        """Step 2, combine all parts"""
        self.clear_scene()
        try:
            # Set up the import settings
            bpy.ops.import_scene.fbx(
                filepath=body_arm_path
            )
        except Exception as e:
            print(f'error loading fbx {e}')
        
        armature = self.find_obj('ARMATURE')[0]
        bpy.context.object.show_in_front = True
        bpy.context.object.data.show_axes = True
        body_head = self.find_obj('MESH')
        
        # # load cloth object
        # with open(f'{self.input_path}/warp_lst.json', 'r') as f:
        #     cloth_data = json.load(f)
        
        for k, v in cloth_data.items():
            part_dir = f"{self.input_path}/{k.split('/')[-1]}/"
            obj_files = glob.glob(os.path.join(part_dir, "*.obj"))
            for obj_file in obj_files:
                print(f"load {obj_file}")
                bpy.ops.wm.obj_import(filepath=obj_file)
                cloth_obj = bpy.context.object
                cloth_obj.rotation_mode = "XYZ"
                cloth_obj.rotation_euler[0] = 0
                bpy.ops.object.select_all(action='DESELECT')
                cloth_obj.select_set(True)
                bpy.context.view_layer.objects.active = armature
                bpy.ops.object.parent_set(type='ARMATURE_NAME')
                
                # weight_name = f'{self.input_path}/weights_{cloth_obj.name}.npz'
                weight_data = auto_rigging[f'weights_{cloth_obj.name}']
                verts_combined = weight_data['verts']
                faces = weight_data['faces']
                weights = weight_data['weights']
                joint_names = weight_data['joint_names']
                # set blending weights
                bpy.context.view_layer.objects.active = cloth_obj
                for index, v in enumerate(cloth_obj.data.vertices):
                    vert = np.array(cloth_obj.matrix_world @ v.co)
                    dist = np.linalg.norm(vert[None] - verts_combined,axis=-1)
                    idx = np.argmin(dist)
                    vertex_weights = weights[idx]
                    for joint_index, joint_weight in enumerate(vertex_weights):
                        if joint_weight > 0.0:
                            vg = cloth_obj.vertex_groups[joint_names[joint_index]]
                            vg.add([index], joint_weight, "REPLACE")          
        

def main(input_path, is_local=False, is_auto_weight=False):
    myclass = MyClass(input_path, is_local)
    utils.delete_all()
    if not is_auto_weight:
        myclass.import_part()
        base_func.set_mesh_original()

        bpy.ops.import_scene.fbx(filepath=os.path.join(myclass.body_path, "auto_rig/armature_mesh.fbx"))
        for mesh in bpy.context.scene.objects:
            if mesh.type == 'MESH':
                if "head" in mesh.name:
                    mesh.name = names["head"]
                elif "body" in mesh.name:
                    mesh.name = names["body"]
        bpy.ops.import_scene.fbx(filepath=os.path.join(myclass.body_path, "auto_rig/smplx_armature_mesh.fbx"))

        myclass.split_faces_part()
        if "shoe" in myclass.part_names:
            myclass.split_faces_shoe()
        if "hair" in myclass.part_names:
            myclass.split_faces_head()

        myclass.skinning_part()
    else:
        myclass.auto_weight()

    myclass.export()
    
if __name__ == '__main__':
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')      

    argv = sys.argv
    raw_paths = argv[argv.index("--") + 1:]
    input_path = raw_paths[0]
    main(input_path)
        