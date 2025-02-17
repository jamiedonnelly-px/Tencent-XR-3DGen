import bpy, json, mathutils
 
bone_json_str = """
{
"Spine":"spine1",
"Spine1":"spine2",
"Spine2":"spine3",
"Neck":"neck",
"Head":"head",

"LeftShoulder":"clavicle_l",
"LeftArm":"upperarm_l",
"LeftForeArm":"lowerarm_l",
"LeftHand":"hand_l",

"RightShoulder":"clavicle_r",
"RightArm":"upperarm_r",
"RightForeArm":"lowerarm_r",
"RightHand":"hand_r",

"Hips":"pelvis",

"LeftUpLeg":"thigh_l",
"LeftLeg":"calf_l",
"LeftFoot":"foot_l",
"LeftToeEnd":"toes_l",

"RightUpLeg":"thigh_r",
"RightLeg":"calf_r",
"RightFoot":"foot_r",
"RightToeEnd":"toes_r",

"LeftHandThumb1":"left_thumb1",
"LeftHandThumb2":"left_thumb2",
"LeftHandThumb3":"left_thumb3",
"LeftHandThumb4":"left_thumb3_end",
"LeftHandIndex1":"left_index1",
"LeftHandIndex2":"left_index2",
"LeftHandIndex3":"left_index3",
"LeftHandIndex4":"left_index3_end",
"LeftHandMiddle1":"left_middle1",
"LeftHandMiddle2":"left_middle2",
"LeftHandMiddle3":"left_middle3",
"LeftHandMiddle4":"left_middle3_end",
"LeftHandRing1":"left_ring1",
"LeftHandRing2":"left_ring2",
"LeftHandRing3":"left_ring3",
"LeftHandRing4":"left_ring3_end",
"LeftHandPinky0":"",
"LeftHandPinky1":"left_pinky1",
"LeftHandPinky2":"left_pinky2",
"LeftHandPinky3":"left_pinky3",
"LeftHandPinky4":"left_pinky3_end",


"RightHandThumb1":"right_thumb1",
"RightHandThumb2":"right_thumb2",
"RightHandThumb3":"right_thumb3",
"RightHandThumb4":"right_thumb3_end",
"RightHandIndex1":"right_index1",
"RightHandIndex2":"right_index2",
"RightHandIndex3":"right_index3",
"RightHandIndex4":"right_index3_end",
"RightHandMiddle1":"right_middle1",
"RightHandMiddle2":"right_middle2",
"RightHandMiddle3":"right_middle3",
"RightHandMiddle4":"right_middle3_end",
"RightHandRing1":"right_ring1",
"RightHandRing2":"right_ring2",
"RightHandRing3":"right_ring3",
"RightHandRing4":"right_ring3_end",
"RightHandPinky0":"",
"RightHandPinky1":"right_pinky1",
"RightHandPinky2":"right_pinky2",
"RightHandPinky3":"right_pinky3",
"RightHandPinky4":"right_pinky3_end"
}
"""

mesh_json_str = """
{
    "body_mesh":{
        "mesh_name":"SM_Body",
        "material_name":"M_Body_Nude"
    },
    "head_mesh":{
        "mesh_name":"SM_Head",
        "material_name":"M_Head_Nude"
    },
    "hair":{
        "mesh_name":"SM_Hair",
        "material_name":"M_Hair"
    },
    "top":{
        "mesh_name":"SM_Top",
        "material_name":"M_Top"
    },
    "trousers":{
        "mesh_name":"SM_Bottom",
        "material_name":"M_Bottom"
    },
    "shoe_left":{
        "mesh_name":"SM_Shoe_Left",
        "material_name":"M_Shoe_Left"
    },
    "shoe_right":{
        "mesh_name":"SM_Shoe_Right",
        "material_name":"M_Shoe_Right"
    }
}
"""
  
def rename_all_bones():
    for armature in bpy.context.scene.objects:
        if armature.type == "ARMATURE":
            armature.select_set(True)
            bpy.context.view_layer.objects.active = armature 

            bpy.ops.object.mode_set(mode='EDIT')
            armature.name = "Root"
            armature.data.name = "Amature"
            
            # 删除末端重复骨骼
            for bone in armature.data.edit_bones:
                if '_end_end' in bone.name or 'root' in bone.name:
                    armature.data.edit_bones.remove(bone)

            # 规范化坐标轴
            # for edit_bone in armature.data.edit_bones:
            #     head_loc = edit_bone.head
            #     edit_bone.tail = head_loc + mathutils.Vector((0, 0, 0.1))
                    
            # 遍历所有的骨头并修复roll角度
            for bone in armature.data.edit_bones:
                bone.roll = 0

            bpy.ops.object.mode_set(mode='OBJECT')
            
            # rename
            names = json.loads(bone_json_str)
            for i in names:
                for bone in armature.data.bones:
                    if bone.name == names[i]:
                        bone.name = i

            

def rename_all_mesh():
    mesh_names = json.loads(mesh_json_str)
    for mesh in bpy.context.scene.objects:
        if mesh.type == "MESH":
            for i in mesh_names:
                if mesh.name == i:
                    mesh.name = mesh_names[i]["mesh_name"]
                    mesh.data.name = mesh_names[i]["mesh_name"]
                    if mesh.data.materials:
                        mesh.data.materials[0].name = mesh_names[i]["material_name"]
                    else:
                        new_mat = bpy.data.materials.new(name=mesh_names[i]["material_name"])
                        mesh.data.materials.append(new_mat)

def rename_openxr():
    rename_all_bones()
    rename_all_mesh()
