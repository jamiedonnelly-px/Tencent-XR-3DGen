import argparse
import json
import sys
import time

import bmesh
import bpy
import numpy as np
import trimesh


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def load_mesh(mesh_path: str, z_up=False):
    bpy.ops.object.select_all(action='DESELECT')
    version_info = bpy.app.version
    print(mesh_path)
    print(version_info)
    if version_info[0] > 2:
        if z_up:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='Y', up_axis='Z')
        else:
            bpy.ops.wm.obj_import(filepath=mesh_path,
                                  forward_axis='NEGATIVE_Z', up_axis='Y')
    else:
        bpy.ops.import_scene.obj(
            filepath=mesh_path, axis_forward='-Z', axis_up='Y')
    meshes = []
    for ind, obj in enumerate(bpy.context.selected_objects):
        if obj.type == 'MESH':
            meshes.append(obj)
    return meshes


def export_mesh_obj(mesh, mesh_path, path_mode='STRIP', global_scale=1, z_up=False):
    print("export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    if z_up:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='Y', up_axis='Z',
                              global_scale=global_scale)
    else:
        bpy.ops.wm.obj_export(filepath=mesh_path,
                              path_mode=path_mode,
                              forward_axis='NEGATIVE_Z', up_axis='Y',
                              global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh


def join_list_of_mesh(mesh_list):
    assert len(mesh_list) > 0
    if len(mesh_list) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for ind, obj in enumerate(mesh_list):
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
        joint_mesh = bpy.context.object
    else:
        joint_mesh = mesh_list[0]
    return joint_mesh


def read_mesh_to_ndarray(mesh, mode="Edit"):
    ''' read the vert coordinate of a deformed mesh
    :param mesh: mesh object
    :return: numpy array of the mesh
    '''
    assert mode in ["Edit", "object"]

    if mode == "object":
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(mesh, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        faces = [[v.index for v in face.verts] for face in bm.faces]
        faces = np.asarray(faces, dtype=np.int32)
        bm.free()
    elif mode == "Edit":
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        faces = [[v.index for v in face.verts] for face in bm.faces]
        faces = np.asarray(faces, dtype=np.int32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co, faces


def triangulate(the_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    the_mesh.modifiers.new("triangulate", "TRIANGULATE")
    bpy.ops.object.convert(target='MESH')  # bake modifier to mesh
    return the_mesh


def copy_uv(mesh_1, mesh_2):
    bpy.ops.object.select_all(action='DESELECT')
    uv_map_names = []
    new_uvmap_prefix = 'projected_uv_'
    # new_uv_map = mesh_1.uv_layers.new(name=new_uvmap_name)
    # new_uv_map.active = True

    mesh_1.select_set(True)
    mesh_2.select_set(True)
    bpy.context.view_layer.objects.active = mesh_2
    for uv in mesh_2.data.uv_layers:
        # set the uv in obj_b to active
        uv.active = True
        # create a new uv in obj_a
        new_uvmap_name = new_uvmap_prefix + uv.name
        uv_map_names.append(new_uvmap_name)
        new_uv = mesh_1.data.uv_layers.new(name=new_uvmap_name)
        # set the uv in obj_a as active
        new_uv.active = True
        bpy.ops.object.join_uvs()

    return uv_map_names


def delete_vertices(the_mesh, indices_to_delete):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    vert_index_uv_map = {}
    for face in the_mesh.data.polygons:
        for vert_idx, loop_idx in zip(face.vertices, face.loop_indices):
            uv_coords = the_mesh.data.uv_layers.active.data[loop_idx].uv
            if vert_idx not in indices_to_delete:
                vert_index_uv_map[vert_idx] = uv_coords

    new_mesh_data = bpy.data.meshes.new(name="new_mesh")
    new_verts = []
    new_uvs = []
    for i, vertex in enumerate(the_mesh.data.vertices):
        if i not in indices_to_delete:
            new_verts.append(vertex.co)
            new_uvs.append(the_mesh.data.uv_layers.active.data[i].uv)

    new_faces = []
    for f in the_mesh.data.polygons:
        if all(i not in indices_set for i in f.vertices):
            new_faces.append(f.vertices)

    new_mesh_data.from_pydata(new_verts, [], new_faces)
    new_mesh_data.update()
    new_obj = bpy.data.objects.new("new_object", new_mesh_data)
    bpy.context.collection.objects.link(new_obj)

    bpy.ops.object.mode_set(mode='OBJECT')
    return the_mesh


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Parse mesh info start. Local time is %s" % (local_time_str))

    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--input_mesh_path', type=str, default="",
                        help='path of input mesh file')
    parser.add_argument('--input_smpl_npz_path', type=str, default="",
                        help='path of input smpl npz file path')
    parser.add_argument('--output_mesh_path', type=str, default="",
                        help='path of output mesh file')
    parser.add_argument('--smpl_segmentation_path', type=str, default="",
                        help='path of input smpl parts segmentation file')
    parser.add_argument('--radius', type=float, default=0.1,
                        help='path of input smpl parts segmentation file')

    args = parser.parse_args(raw_argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    input_mesh_path = args.input_mesh_path
    input_smpl_npz_path = args.input_smpl_npz_path
    output_mesh_path = args.output_mesh_path
    smpl_segmentation_path = args.smpl_segmentation_path
    radius = args.radius

    original_mesh = load_mesh(input_mesh_path)
    joint_mesh = join_list_of_mesh(original_mesh)
    joint_mesh = triangulate(joint_mesh)

    smpl_data_struct = np.load(input_smpl_npz_path)
    smpl_faces = smpl_data_struct["faces"]
    smpl_posed_verts = smpl_data_struct["posed_verts"].squeeze(axis=0)

    smpl_segmentation_info = read_json(smpl_segmentation_path)
    right_hand_point_index = smpl_segmentation_info["rightHand"]
    left_hand_point_index = smpl_segmentation_info["leftHand"]
    right_forearm_point_index = smpl_segmentation_info["rightForeArm"]
    left_forearm_point_index = smpl_segmentation_info["leftForeArm"]
    right_arm_point_index = smpl_segmentation_info["rightArm"]
    left_arm_point_index = smpl_segmentation_info["leftArm"]
    all_arm_index = right_arm_point_index + right_hand_point_index + right_forearm_point_index
    all_arm_index = all_arm_index + left_arm_point_index + left_forearm_point_index + left_hand_point_index

    selected_vertices = smpl_posed_verts[all_arm_index]
    radis_array = np.ones((len(all_arm_index)), dtype=np.float32) * radius

    joint_verts, joint_faces = read_mesh_to_ndarray(joint_mesh)
    joint_tri_mesh = trimesh.Trimesh(vertices=joint_verts, faces=joint_faces)

    internal_rotation = trimesh.transformations.euler_matrix(np.pi / 2, 0.0, 0.0, 'rxyz')
    joint_tri_mesh.apply_transform(internal_rotation)

    close_points_indices = joint_tri_mesh.kdtree.query_ball_point(selected_vertices, radis_array)
    indices_to_delete = close_points_indices.flatten().tolist()
    indices_set = set()
    for index_list in indices_to_delete:
        for index in index_list:
            indices_set.add(index)

    deleted_mesh = delete_vertices(joint_mesh, indices_set)

    export_mesh_obj(joint_mesh, output_mesh_path, path_mode='COPY')
