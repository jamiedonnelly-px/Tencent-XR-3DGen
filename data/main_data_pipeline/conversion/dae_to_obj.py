import argparse
import os
import sys

import bmesh
import bpy
import numpy as np

weapon = ["weapon", "Weapon"]


def read_mesh_to_ndarray(mesh, mode="Edit"):
    ''' read the vert coordinate of a deformed mesh
    :param mesh: mesh object
    :return: numpy array of the mesh
    '''
    assert mode in ["edit", "object"]

    if mode == "object":
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(mesh, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        # faces = [[v.index for v in face.verts] for face in bm.faces]
        # faces = np.asarray(faces, dtype=np.int32)
        bm.free()
    elif mode == "edit":
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray(mverts_co, dtype=np.float32)
        # faces = [[v.index for v in face.verts] for face in bm.faces]
        # faces = np.asarray(faces, dtype=np.int32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co, None


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


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--output_fullpath', type=str,
                        default="", help='render result output folder')
    parser.add_argument('--apply_pose_toggle', action='store_true',
                        help='force toggle pose mode of the model')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')
    parser.add_argument('--force_z_up', action='store_true',
                        help='force use z/y axis in obj exporting')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_fullpath = args.output_fullpath
    output_folder = os.path.split(output_fullpath)[0]

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.wm.collada_import(filepath=mesh_path)

    if args.apply_pose_toggle:
        try:
            # switch character to rest mode, i.e. A-pose in most case
            bpy.ops.object.posemode_toggle()
            bpy.ops.pose.select_all(action='SELECT')
            bpy.ops.pose.loc_clear()
            bpy.ops.pose.rot_clear()
            bpy.ops.pose.scale_clear()
            bpy.ops.object.posemode_toggle()
        except:
            print('posemode_toggle failed')

    bpy.ops.object.select_all(action='DESELECT')
    meshes = []
    size_meshes = []
    for ind, obj in enumerate(bpy.context.scene.objects):
        if obj.type == 'MESH':
            if any(wp in obj.name for wp in weapon):
                obj.select_set(state=True)
                bpy.ops.object.delete()
            else:
                meshes.append(obj)

    if len(meshes) < 1:
        print("No avatar found in model.....")
        exit(-1)

    joint_mesh = join_list_of_mesh(meshes)
    print("Convert dae mesh from %s to %s" % (mesh_path, output_fullpath))

    if args.copy_texture:
        export_mesh_obj(joint_mesh, output_fullpath,
                        path_mode='COPY', z_up=args.force_z_up)
    else:
        export_mesh_obj(joint_mesh, output_fullpath,
                        path_mode='RELATIVE', z_up=args.force_z_up)
