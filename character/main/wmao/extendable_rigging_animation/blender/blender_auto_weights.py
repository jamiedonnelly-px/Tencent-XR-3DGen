import bpy
import numpy as np
import mathutils
from mathutils import Vector
import math
import copy
import argparse
import os
import sys

def clear_scene():
    # Ensure we are in Object mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


# Function to find the armature in the scene
def find_obj(type="ARMATURE", inverse=False):
    objs = []
    for obj in bpy.context.scene.objects:
        if not inverse and obj.type == type:
            objs.append(obj)
        elif inverse and obj.type != type:
            objs.append(obj)
    return objs



if __name__ == "__main__":
    
    argv = sys.argv
    arg_idx = argv.index("--")
    path = argv[arg_idx + 1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_npz', type=str, default="/Users/weimao/Documents/test_lbs/output/elephant3/rigging/verts_joints.npz")
    parser.add_argument('--out_file', type=str, default='/Users/weimao/Documents/test_lbs/output/elephant3/rigging/blender_auto_weights.npz')
    # args = parser.parse_args()
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    clear_scene()
    # os.makedirs(args.out_dir,exist_ok=True)
    data = np.load(args.data_npz)
    
    joints = data['joints']
    parents = data['parents']
    parent_names = data['parent_names']
    joint_names = data['joint_names']
    verts = data['verts']
    faces = data['faces']
    
    # create mesh
    vertices = verts.tolist()
    faces = faces.tolist()
#    vertices = [Vector(v) for v in vertices]
#    faces = [Vector(f) for f in faces]
    mesh = bpy.data.meshes.new(name='Mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    # Create a new object with the mesh
    obj = bpy.data.objects.new(name='Mesh', object_data=mesh)

    # Link the object to the current scene collection
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # combine duplicated vertices
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.context.scene.cursor.location[0] = 0
    bpy.context.scene.cursor.location[1] = 0
    bpy.context.scene.cursor.location[2] = 0

    # create armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = 'Armature'

    # Access the edit mode of the armature
    bpy.ops.object.mode_set(mode='EDIT')
    # remove the default bone
    bone_to_delete = armature.data.edit_bones.get('Bone')
    if bone_to_delete:
        armature.data.edit_bones.remove(bone_to_delete)

#    assert False
    # Create bones
    bones = [[]]
    for ji, jn in enumerate(joint_names.tolist()):
        # assume the first joint is useless for blending weights
        if ji == 0:
            continue
        print('process ', jn)
        ch = []
        for pi, p in enumerate(parents):
            if p == ji:
                ch.append(pi)
        if len(ch) > 0:
            pos_ch = joints[ch].mean(axis=0)
        pos = joints[ji]
        
        bone = armature.data.edit_bones.new(jn)
        bone.head = pos
        if ji > 1:
            bone.parent = bones[parents[ji]]
            bone.use_connect = True
        # Set tail of the previous bone to be the head of the current bone
        if len(ch) > 0:
            bone.tail = pos_ch
        else:
#            0.1 * (bone.head - bone.parent.head)
            bone.tail = bone.head +  0.4 * (bone.head - bone.parent.head)
        bones.append(bone)
    
    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    armature = find_obj(type="ARMATURE", inverse=False)[0]
    bpy.context.object.show_in_front = True
    mesh = find_obj(type="MESH")[0]
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature 
    
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    
    # get blending weights
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    weights = []
    nb = len(joint_names)
    obj = mesh
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
    
    np.savez_compressed(f'{args.out_file}',verts=vs,weights=w)