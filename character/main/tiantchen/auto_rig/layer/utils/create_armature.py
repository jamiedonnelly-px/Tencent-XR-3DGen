# Blender
import os, sys, bpy, math, json, torch, mathutils
import numpy as np
from math import radians
from pathlib import Path
from smplx.joint_names import JOINT_NAMES
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from hair_color_change import change_hair, material_with_Kd
from utils_blender import utils, base_func, fix_mesh_transparency, names

SMPLX_JOINT_NAMES = ['thigh_l','thigh_r', 'spine1', 'calf_l','calf_r', 'spine2', 'foot_l','foot_r', 'spine3', 'toes_l','toes_r', 'neck', 'clavicle_l','clavicle_r', 'head', 'upperarm_l','upperarm_r', 'lowerarm_l', 'lowerarm_r', 'hand_l','hand_r']
skip_index = [3,4,  11,14,  2,5,8] # calf, neck head, spine, clavicle, 12,13
shoulder_and_collar_index = [12,13,15,16]

root_dir = "/aigc_cfs/tinatchen/layer/bodyfit"
sys.path.append(os.path.join(root_dir, "delta"))
sys.path.append(os.path.join(root_dir, "smpl_weights"))
from lib.configs.config_vroid import get_cfg_defaults
from lib.smpl_w_scale import SMPL_with_scale

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = mathutils.Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()
    quat = mathutils.Quaternion(axis, angle_rad)
    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    return

def set_pose_from_rodrigues_inv(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = mathutils.Vector((rodrigues[0], rodrigues[1], rodrigues[2]))

    angle_rad = rod.length
    axis = rod.normalized()
    quat = mathutils.Quaternion(axis, angle_rad)
    quat_inv = quat.inverted()
    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat_inv            
    return

def bind_mesh_armature(obj_mesh, armature):
    bpy.ops.object.select_all(action='DESELECT') 
    obj_mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    obj_mesh.parent = armature
    obj_mesh.parent_type = 'ARMATURE'
    armature_modifier = obj_mesh.modifiers.new(name="Armature", type="ARMATURE")
    armature_modifier.object = armature
    print(obj_mesh, armature)

class MyClass:
    def __init__(self, input_path):
        self.SMPLX_JOINT_NAMES = SMPLX_JOINT_NAMES
        self.NUM_SMPLX_JOINTS = len(self.SMPLX_JOINT_NAMES)

        cfg = get_cfg_defaults()

        smplxs = SMPL_with_scale(cfg).to(torch.device(0))
        self.body_path = input_path
        self.body_obj_path = os.path.join(self.body_path, "naked")
        self.body_type = self.body_path.split('/')[-1].split('_')[0]
        print("self.body_type", self.body_type)

        smplx_model_data =  torch.load( os.path.join(self.body_path, "smplx_and_offset.npz" ))
        self.body_pose = torch.tensor(smplx_model_data['body_pose'])
        faces,  vertices, init_joints, joints, lbs_weights, parents, J_regressor = smplxs.forward(smplx_model_data)
        self.faces = faces
        self.vertices = vertices
        self.init_joints = init_joints
        self.joints = joints
        self.lbs_weights = lbs_weights
        self.parents = parents
        self.J_regressor = J_regressor
        
        smplx_data = json.load(open(os.path.join(root_dir, "smpl_weights/smplx_2020/smplx_vert_segmentation_pro.json")))

        # smplx 头部idx
        self.smplx_head_idx = smplx_data["pro"]["head"]
        self.smplx_left_verts_idx = smplx_data["pro"]["left"]
        self.smplx_right_verts_idx = smplx_data["pro"]["right"]
        
        self.angle = math.radians(-90)
        self.rotation = mathutils.Matrix.Rotation(self.angle, 3, 'X')

    def import_body(self):
        if  self.body_type == "readyplayerme":
            utils.import_file(os.path.join(self.body_obj_path, "head_body.glb"))
        else:
            utils.import_file(os.path.join(self.body_obj_path, "head_body.obj"))

        for mesh in bpy.context.scene.objects:
            if mesh.type == 'MESH':
                if "head" in mesh.name:
                    mesh.name = names["head"]
                if "body" in mesh.name:
                    mesh.name = names["body"]
                    
    def create_skeletal_mesh(self):
        (vertices, faces) = self.blender_mesh_from_model()
        name = "SMPLX-shapes-female"
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new(names["smplx_mesh"], mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.select_set(True)

        # Create armature
        armature_object =self.create_armature(names["smplx_armature"])

        # Bind mesh to armature (skinning)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_object
        bpy.ops.object.parent_set(type="ARMATURE_NAME") # Create empty vertex groups

        # Remove root vertex group
        bpy.context.view_layer.objects.active = obj
        obj.vertex_groups.active_index = 0
        bpy.ops.object.vertex_group_remove()

        # Set skin weights
        bpy.context.view_layer.objects.active = obj
        for index, vertex_weights in enumerate(self.lbs_weights):
            for joint_index, joint_weight in enumerate(vertex_weights):
                if joint_weight > 0.0:
                    # Get vertex group for joint and add vertex index with weight
                    if joint_index < 22 and joint_index > 0:
                        vg = obj.vertex_groups[self.SMPLX_JOINT_NAMES[joint_index-1]]
                    else:
                        vg = obj.vertex_groups[JOINT_NAMES[joint_index]]
                    vg.add([index], joint_weight, "REPLACE")

        # Use smooth normals
        obj.select_set(True)
        bpy.ops.object.shade_smooth()

        # Armature is now the main object and skinned mesh is child of it
        return armature_object

    def create_armature(self, armature_name, trans=True):
        joints = self.blender_joints_from_model(trans)
        # Create armature
        bpy.ops.object.armature_add()
        armature_object = bpy.context.selected_objects[0]
        armature_object.name = armature_name
        armature_object.data.name = "SMPLX-armature-female"
        armature_object.location = (0, 0, 0)
        armature = armature_object.data
        armature.bones[0].name = "root"
        
        num_joints = len(joints)
        bpy.ops.object.mode_set(mode="EDIT")
        joint_name = 'pelvis'
        bpy.ops.armature.bone_primitive_add(name=joint_name)
        
        for index in range(1, 22):
            joint_name = self.SMPLX_JOINT_NAMES[index-1]
            bpy.ops.armature.bone_primitive_add(name=joint_name)
        for index in range(22, num_joints):
            joint_name = JOINT_NAMES[index]
            bpy.ops.armature.bone_primitive_add(name=joint_name)
        for bone in armature.edit_bones:
            bone.head = (0.0, 0.0, 0.0)
            bone.tail = (0.0, 0.0, 0.1)
        for index in range(len(self.parents)):
            if index == 0:
                parent_index = -1 
            else:
                parent_index = self.parents[index].item() #model.kintree_table[0][index]
            armature.edit_bones[index + 1].parent = armature.edit_bones[parent_index + 1]
            bone_start = self.rotation @ mathutils.Vector(joints[index])
            if index < 22 and index > 0:
                armature.edit_bones[self.SMPLX_JOINT_NAMES[index-1]].translate(bone_start)
            else:
                armature.edit_bones[JOINT_NAMES[index]].translate(bone_start)
                

        bpy.ops.object.mode_set(mode="OBJECT")
        return armature_object

    def create_smplx_armature_mesh(self):
        (vertices, faces) = self.blender_mesh_from_model()
        name = "SMPLX-shapes-female"
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new(names["smplx_mesh"], mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.select_set(True)

        # Create armature
        armature_object =self.create_armature(names["smplx_armature"])

        # Bind mesh to armature (skinning)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_object
        bpy.ops.object.parent_set(type="ARMATURE_NAME") # Create empty vertex groups

        # Remove root vertex group
        bpy.context.view_layer.objects.active = obj
        obj.vertex_groups.active_index = 0
        bpy.ops.object.vertex_group_remove()

        # Set skin weights
        bpy.context.view_layer.objects.active = obj
        lbs_weights = self.lbs_weights
        for index, vertex_weights in enumerate(lbs_weights):
            for joint_index, joint_weight in enumerate(vertex_weights):
                if joint_weight > 0.0:
                    # Get vertex group for joint and add vertex index with weight
                    if joint_index < 22 and joint_index > 0:
                        vg = obj.vertex_groups[self.SMPLX_JOINT_NAMES[joint_index-1]]
                    else:
                        vg = obj.vertex_groups[JOINT_NAMES[joint_index]]
                    vg.add([index], joint_weight, "REPLACE")

        # Use smooth normals
        obj.select_set(True)
        bpy.ops.object.shade_smooth()

        # Armature is now the main object and skinned mesh is child of it
        return armature_object
    
    # Get Blender mesh data from model for given betas and expression
    def blender_mesh_from_model(self):
        vertices = self.vertices.detach().cpu().numpy().squeeze()   
        rotate = np.array(self.rotation) 
        vertices = np.dot(vertices, rotate.T)
        faces = self.faces
        vertices_blender = []
        faces_blender = []
        for v in vertices:
            vertices_blender.append((v[0], -v[2], v[1]))
        for f in faces:
            faces_blender.append((f[0], f[1], f[2]))
        return (vertices_blender, faces_blender)

    # Get Blender joint locations from model for given betas
    def blender_joints_from_model(self, trans=True):
        if trans:
            joints = self.joints.detach().cpu().numpy().squeeze()
        else:
            joints = self.init_joints.detach().cpu().numpy().squeeze()
        num_joints = self.J_regressor.shape[0]
        joints_blender = []
        for joint_index in range(num_joints):
            j = joints[joint_index]
            joints_blender.append((j[0], -j[2], j[1]))
        return joints_blender     

    def skinning_weight_knn(self, armature_object, mesh_object, mesh_smplx_object, mesh_name):
        mesh_object.vertex_groups.clear()
        bones = armature_object.data.bones
        # create group
        for bone in bones:
           vertex_group = mesh_object.vertex_groups.new(name=bone.name)
           
        mesh_vertices_list = [v.co for v in mesh_object.data.vertices]
        mesh_vertices_array = np.array(mesh_vertices_list)
               
        smplx_vertices_list = [v.co for v in mesh_smplx_object.data.vertices]
        smplx_vertices_array = np.array(smplx_vertices_list)

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(smplx_vertices_array)
        nearest_indices = knn.kneighbors(mesh_vertices_array, return_distance=False)    

        for i in range(len(nearest_indices)):
            min_index = nearest_indices[i][0] 

            vertex_in_groups = []
            for vertex_group in mesh_smplx_object.vertex_groups:
                try:
                    weight = vertex_group.weight(min_index)
                    vertex_in_groups.append((vertex_group.name, weight))
                except:
                    pass
            for vertex_in_group in vertex_in_groups:
                vertex_group = mesh_object.vertex_groups.get(vertex_in_group[0])
                vertex_group.add([i], vertex_in_group[1], 'REPLACE')
                    
                    
    def T_to_transform(self, armature):
        bpy.ops.object.mode_set(mode='EDIT')
        body_pose_reshape = self.body_pose.reshape(self.NUM_SMPLX_JOINTS, 3)
        for index in range(self.NUM_SMPLX_JOINTS):
            if index in skip_index:
                continue
            pose_rodrigues = body_pose_reshape[index]
            if index in shoulder_and_collar_index:
                    pose_rodrigues[0] = 0
                    pose_rodrigues[1] = 0
            bone_name = self.SMPLX_JOINT_NAMES[index] # body pose starts with left_hip
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)  
        bpy.ops.object.mode_set(mode='OBJECT')

        # fixed pose
        bpy.ops.object.select_all(action='DESELECT') 
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.armature_apply(selected=False)
        bpy.context.view_layer.update()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.context.view_layer.update() 
        
    def transform_to_T(self, mesh_names, armature): 
        if self.body_type != "timer":     
            bpy.ops.object.mode_set(mode='EDIT')
            body_pose = self.body_pose.reshape(self.NUM_SMPLX_JOINTS, 3)
            for index in range(self.NUM_SMPLX_JOINTS):
                if index in skip_index:
                    continue
                pose_rodrigues = body_pose[index]
                if index in shoulder_and_collar_index:
                    pose_rodrigues[0] = 0
                    pose_rodrigues[1] = 0
                bone_name = self.SMPLX_JOINT_NAMES[index] # body pose starts with left_hip
                set_pose_from_rodrigues_inv(armature, bone_name, pose_rodrigues)
            bpy.ops.object.mode_set(mode='OBJECT')
        
        for mesh_name in mesh_names:
            bpy.ops.object.select_all(action='DESELECT') 
            mesh = bpy.data.objects[mesh_name]
            mesh.select_set(True)
            
            if mesh.data.shape_keys is not None:
                for key_block in mesh.data.shape_keys.key_blocks:
                    mesh.shape_key_remove(key_block)
            for modifier in mesh.modifiers:
                bpy.ops.object.modifier_apply({"object": mesh}, modifier=modifier.name)
            bpy.context.view_layer.update()
            
        bpy.ops.object.select_all(action='DESELECT') 
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.armature_apply(selected=False)
        bpy.context.view_layer.update()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        
        # for mesh_name in mesh_names:
        utils.select(mesh_names)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.context.view_layer.update()
        
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action='DESELECT') 

        # Bind mesh to armature (skinning)
        for mesh_name in mesh_names:
            bpy.ops.object.select_all(action='DESELECT') 
            mesh = bpy.data.objects[mesh_name]
            mesh.select_set(True)
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.parent_set(type="ARMATURE_NAME")  

def make_json(json_path):
    if os.path.exists(json_path):
        return
    data = { "head": { "delete": [], "undelete": []}, 
    "body": { "delete": [], "undelete": [], "foot":[], 
             "loop": {"leg_l": [], "leg_r": [], "neck": [], "arm_l": [], "arm_r": []} } }

    json_data = json.dumps(data, indent=4)

    with open(json_path, 'w') as f:
        f.write(json_data)

def main(path):
    utils.delete_all()
                
    myclass = MyClass(path)

    myclass.import_body()
    base_func.set_mesh_original()
    
    os.makedirs(os.path.join(myclass.body_path, "auto_rig"), exist_ok=True)

    make_json(os.path.join(myclass.body_path, "auto_rig/mask_faces.json"))

    myclass.create_smplx_armature_mesh()

    utils.export_fbx(os.path.join(myclass.body_path, "auto_rig/smplx_armature_mesh.fbx"), [names["smplx_armature"], names["smplx_mesh"]])
    
    myclass.create_armature(names["armature"], False)
        
    mesh_smplx_object = bpy.data.objects[names["smplx_mesh"]]
    armature = bpy.data.objects[names["armature"]]
    
    # myclass.T_to_transform(armature)
    mesh_names = []
    for mesh in bpy.context.scene.objects:
        if mesh.type == 'MESH':
            if "SMPLX" not in mesh.name:
                if "head" in mesh.name:
                    mesh.name = names["head"]
                elif "body" in mesh.name:
                    mesh.name = names["body"]
                print("mesh name: ", mesh.name)
                for modifier in mesh.modifiers:
                    bpy.ops.object.modifier_remove({"object":mesh}, modifier=modifier.name)
                for vertex_group in mesh.vertex_groups:
                    mesh.vertex_groups.remove(vertex_group)
                mesh_names.append(mesh.name)
                bind_mesh_armature(mesh, armature)
                myclass.skinning_weight_knn(armature, mesh, mesh_smplx_object, mesh.name)
                
    # myclass.transform_to_T(mesh_names, armature)
                
    fix_mesh_transparency.run(os.path.join(myclass.body_path, "auto_rig"))
    utils.export_fbx(os.path.join(myclass.body_path, "auto_rig/armature_mesh.fbx"),[names["armature"], names["body"], names["head"]])

if __name__ == "__main__":   
    argv = sys.argv
    path = argv[argv.index("--") + 1]
    main(path) 
    
    
    

