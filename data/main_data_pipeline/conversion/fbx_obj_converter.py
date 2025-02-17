import argparse
import os
import shutil
import sys
import time

import bpy


# weapon = ["weapon", "Weapon"]

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


def copy_image_textures(object, new_image_folder):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree
                for node in node_tree.nodes:
                    if node.type == 'TEX_IMAGE':
                        image_path = node.image.filepath
                        image_filename = os.path.split(image_path)[1]
                        new_image_path = os.path.join(
                            new_image_folder, image_filename)
                        shutil.copyfile(image_path, new_image_path)


if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='File converter.')
    parser.add_argument('--mesh_path', type=str,
                        help='path to mesh to be rendered')
    parser.add_argument('--output_fullpath', type=str,
                        default="", help='render result output folder')
    parser.add_argument('--force_better_fbx', action='store_true',
                        help='force to use better fbx as import plugin')
    parser.add_argument('--apply_pose_toggle', action='store_true',
                        help='force toggle pose mode of the model')
    parser.add_argument('--copy_texture', action='store_true',
                        help='copy original texture file to new folder')
    parser.add_argument('--force_z_up', action='store_true',
                        help='force use z/y axis in obj exporting')
    args = parser.parse_args(raw_argv)

    mesh_path = args.mesh_path
    output_fullpath = args.output_fullpath
    copy_texture = args.copy_texture
    force_better_fbx = args.force_better_fbx
    apply_pose_toggle = args.apply_pose_toggle
    output_folder = os.path.split(output_fullpath)[0]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    blend_folder = os.path.join(output_folder, "blend")
    if not os.path.exists(blend_folder):
        os.mkdir(blend_folder)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    if force_better_fbx:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # print(bpy.ops.preferences.addon_expand(module="better_fbx"))

        # addon_path = os.path.join(os.path.dirname(
        #     os.path.abspath(__file__)), '../addons/better_fbx.zip')
        # bpy.ops.preferences.addon_install(overwrite=True, target='DEFAULT', filepath=addon_path,
        #                                   filter_folder=True, filter_python=False, filter_glob="*.py;*.zip")
        bpy.ops.preferences.addon_enable(module="better_fbx")
        bpy.ops.better_import.fbx(filepath=mesh_path, use_optimize_for_blender=False,
                                  use_auto_bone_orientation=True,
                                  use_reset_mesh_origin=True, use_reset_mesh_rotation=True,
                                  use_detect_deform_bone=True, use_auto_smooth=True,
                                  use_animation=True)

    else:
        bpy.ops.import_scene.fbx(filepath=mesh_path, use_anim=True)

    if apply_pose_toggle:
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
            meshes.append(obj)

    if len(meshes) < 1:
        print("No model found.....")
        exit(-1)

    blend_file_path = os.path.join(blend_folder, "fbx_temp.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path,
                                compress=False,
                                check_existing=False)
    bpy.ops.file.unpack_all(method='WRITE_LOCAL')

    time.sleep(0.1)

    joint_mesh = join_list_of_mesh(meshes)
    print("Export fbx mesh with from %s to %s" % (mesh_path, output_fullpath))

    if copy_texture:
        export_mesh_obj(joint_mesh, output_fullpath,
                        path_mode='COPY', z_up=args.force_z_up)
    else:
        export_mesh_obj(joint_mesh, output_fullpath,
                        path_mode='RELATIVE', z_up=args.force_z_up)
