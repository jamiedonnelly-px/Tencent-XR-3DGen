
import numpy as np
import bpy
import mathutils
from mathutils import Vector, Matrix
import trimesh
import math
import os
import sys
import argparse
import json

def clear_scene(remain_objects=[]):
    # Ensure we are in Object mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    for obj in bpy.context.scene.objects:
        if not obj.visible_get():
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in remain_objects:
        obj.select_set(False)
    bpy.ops.object.delete()
#
#
def align_point_clouds(P, Q):
    """
    Aligns two point clouds, `P` and `Q`, using given correspondences, by solving for the 
    optimal scale, rotation, and translation that minimizes the distance between corresponding 
    points.

    Parameters:
    -----------
    P : numpy.ndarray
        An (n, 3) array representing the first point cloud with `n` points in 3D.
    Q : numpy.ndarray
        An (n, 3) array representing the second point cloud with `n` points in 3D, where each 
        point in `Q` corresponds to a point in `P`.
    Returns:
    --------
    P_aligned : numpy.ndarray
        The transformed version of `P`, aligned to `Q`.
    scale : float
        The scaling factor applied to `P` for optimal alignment with `Q`.
    R : numpy.ndarray
        A (3, 3) rotation matrix that aligns `P` to `Q`.
    translation : numpy.ndarray
        A (3,) translation vector applied after scaling and rotation to align `P` to `Q`.
    Steps:
    ------
    1. Compute the centroids of `P` and `Q`.
    2. Center both point clouds by subtracting their respective centroids.
    3. Calculate the scaling factor that best aligns the centered `P` to the centered `Q`.
    4. Compute the optimal rotation matrix using Singular Value Decomposition (SVD) of the 
       covariance matrix of the centered points.
    5. Adjust for reflection if necessary to ensure a proper rotation.
    6. Apply scaling, rotation, and translation to obtain the aligned points.
    Example:
    --------
    >>> P = np.random.rand(10, 3)  # Replace with actual data
    >>> Q = np.random.rand(10, 3)  # Replace with actual data
    >>> P_aligned, scale, R, translation = align_point_clouds(P, Q)
    >>> print("Aligned Point Cloud:", P_aligned)
    >>> print("Scale:", scale)
    >>> print("Rotation Matrix:\n", R)
    >>> print("Translation Vector:", translation)
    """
    # Step 2: Calculate centroids
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)
    # Step 3: Center the point clouds
    P_prime = P - P_centroid
    Q_prime = Q - Q_centroid
    # Step 4: Compute scale
    scale = np.sum(np.linalg.norm(Q_prime, axis=1) * np.linalg.norm(P_prime, axis=1)) / np.sum(np.linalg.norm(P_prime, axis=1)**2)
    # Step 5: Compute the optimal rotation matrix
    H = P_prime.T @ Q_prime
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure a proper rotation (reflection correction)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Step 6: Apply scaling and combine transformations
    P_aligned = scale * (R @ P_prime.T).T + Q_centroid
    return P_aligned, scale, R, Q_centroid - scale * (R @ P_centroid)


def get_closest_triangle_normal(obj, point):
    """
    Finds the normal of the closest triangle face on a mesh surface to a given point.

    Parameters:
    obj (bpy.types.Object): The mesh object on which to find the closest surface normal.
    point (mathutils.Vector): The 3D point in space where the closest face normal is needed.

    Returns:
    mathutils.Vector: The normal vector of the closest triangle face on the surface.
                      If no face is found, returns None.

    Usage:
    - Ensure the object is a mesh and is in the current scene.
    - Call the function with the mesh object and a 3D point as inputs.
    - The function uses a BVH tree for efficient spatial querying to find the closest face.
    
    Example:
    obj = bpy.context.scene.objects["my_object"]
    point = mathutils.Vector((x, y, z))  # Replace with the actual coordinates
    normal = get_closest_triangle_normal(obj, point)
    print("Closest Triangle Normal:", normal)
    """
    # Ensure the object's geometry is up-to-date
    obj.data.calc_normals_split()
    
    # Create a BVH tree for efficient spatial queries
    bvh_tree = mathutils.bvhtree.BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    
    # Find the closest point on the surface to the specified point
    location, normal, index, distance = bvh_tree.find_nearest(point)
    if index is None:
        print("No nearby face found.")
        return None
    
    # Get the face corresponding to the index and return its normal
    face = obj.data.polygons[index]
    return face.normal




if __name__ == "__main__":
    
    argv = sys.argv
    arg_idx = argv.index("--")
    path = argv[arg_idx + 1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--quest_head_dir', type=str, default="/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/data/quest_head.obj")
    parser.add_argument('--correspondance_npz', type=str, default='/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/test_data/quest_head/correspondance_aligned.npz')
    parser.add_argument('--mp_face_aligned_npz', type=str, default='/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/data/mp_face_aligned_edited.obj')
    parser.add_argument('--mp_face_target_npz', type=str, default='/Users/weimao/Documents/avatar/celebrities/target_name_cartoon/face_mesh.obj')
    parser.add_argument('--mp_face_bary_json', type=str, default='/aigc_cfs_2/weimao/avatar_face_generation/data/quest_head_model/mp_face_bary.json')
    parser.add_argument('--out_file', type=str, default='/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/test_data/quest_target_name.obj')
    # args = parser.parse_args()
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    clear_scene()

    # quest_head = "/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/data/quest_head.obj"
    # correspondence = "/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/test_data/quest_head/correspondance_aligned.npz"
    # mp_face_aligned = "/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/data/mp_face_aligned_edited.obj"
    # target_name = 'taylor'
    # mp_face_target = f"/Users/weimao/Documents/avatar/celebrities/{target_name}_cartoon/face_mesh.obj"
    # out_file = f"/Users/weimao/Documents/avatar/Deformation-Transfer-for-Triangle-Meshes/test_data/quest_{target_name}.obj"
    quest_head = args.quest_head_dir
    correspondence = args.correspondance_npz
    mp_face_aligned = args.mp_face_aligned_npz
    mp_face_target = args.mp_face_target_npz
    out_file = args.out_file
    #
    clear_scene()
    bpy.ops.wm.obj_import(filepath=quest_head)
    head_mesh = bpy.context.active_object
    head_mesh.rotation_mode = 'XYZ'
    head_mesh.rotation_euler[0] = 0   
    #
    corr_data = np.load(correspondence)
    bone_position = corr_data["xyz_quest_of_mp"]
    print('bone position shape', bone_position.shape)
    #
    bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    armature = bpy.context.active_object
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='DESELECT')
    edit_bones = armature.data.edit_bones
    normal = get_closest_triangle_normal(head_mesh, Vector(bone_position[0]))
    edit_bones[0].head = Vector(bone_position[0]) - Vector((0,0,0.002))
    edit_bones[0].tail = Vector(bone_position[0]) + Vector((0,0,0.01))
    edit_bones[0].name = 'marker_000'
    #
    for i in range(1, bone_position.shape[0]):
        new_bone = armature.data.edit_bones.new(f"marker_{i:03d}")
        normal = get_closest_triangle_normal(head_mesh, Vector(bone_position[i]))
        new_bone.head = Vector(bone_position[i]) - Vector((0,0,0.002))
        new_bone.tail = Vector(bone_position[i]) + Vector((0,0,0.01))

    #    
    #    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    head_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    bpy.ops.export_scene.fbx(filepath=f'{os.path.dirname(out_file)}/head_rigged.fbx')  
    #
    # face_template = trimesh.load(mp_face_aligned)
    # vert_template = face_template.vertices
    face_template = np.load(mp_face_aligned)
    vert_template = face_template['verts']
    faces = face_template['faces']
    #
    # face_target = trimesh.load(mp_face_target)
    # vert_target = face_target.vertices
    face_target = np.load(mp_face_target)
    vert_target = face_target['verts']
    vert_target, _, _, _ = align_point_clouds(vert_target, bone_position[:vert_target.shape[0]])

    np.savez_compressed(f'{os.path.dirname(out_file)}/mp_face_target_aligned.npz', verts=vert_target, faces=face_target['faces'], uvs=face_target['uvs'])
    # mesh_align = trimesh.Trimesh(vertices=vert_target,faces=face_template.faces)
    # file_name = os.path.join(os.path.dirname(out_file),"aligned_" + target_name+'.obj')
    # mesh_align.export(file_name)
    with open(args.mp_face_bary_json, 'r') as f:
        bary_dict = json.load(f)
    fids = [int(fid) for fid in list(bary_dict.keys())]
    face_center = (vert_template[faces[fids,0]] + vert_template[faces[fids,1]] + vert_template[faces[fids,2]])/3
    vert_template = np.concatenate([vert_template, face_center],axis=0)
    face_center_target = (vert_target[faces[fids,0]] + vert_target[faces[fids,1]] + vert_target[faces[fids,2]])/3
    vert_target = np.concatenate([vert_target, face_center_target],axis=0)
    print('vert target shape:', vert_target.shape)
    dv = vert_target - vert_template

    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    for i in range(bone_position.shape[0]):
        bone = armature.pose.bones[f'marker_{i:03d}']
        bone.location = bone.matrix.transposed() @ Vector(dv[i])

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    head_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = head_mesh
    bpy.ops.object.modifier_apply(modifier="Armature")
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_clear(type='CLEAR')
    bpy.data.objects.remove(armature, do_unlink=True)
    head_mesh.rotation_euler[0] = math.pi/2

    bpy.ops.wm.obj_export(filepath=out_file, export_selected_objects=True, export_materials=False)
    
