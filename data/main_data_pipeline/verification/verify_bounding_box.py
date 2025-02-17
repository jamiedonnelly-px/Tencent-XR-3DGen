import argparse
import copy
import os
import time

import trimesh


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


def calculate_bounding_box(mesh_path: str):
    if not os.path.exists(mesh_path):
        return None
    Gobj_load = trimesh.load(mesh_path)
    if isinstance(Gobj_load, trimesh.base.Trimesh):
        Gobj = Gobj_load
    else:
        Gobj = as_mesh(Gobj_load)
    aa, bb = Gobj.bounds
    result_aa = copy.deepcopy(aa)
    result_bb = copy.deepcopy(bb)
    return result_aa, result_bb


def calculate_limit(input: float, interval_coeff: float = 0.12, default_interval: float = 0.01):
    interval = input * interval_coeff
    if abs(interval) < default_interval:
        interval = default_interval
    limit1 = input + interval
    limit2 = input - interval
    if limit1 > limit2:
        return limit2, limit1
    else:
        return limit1, limit2


def compare_aabb(candidate, target, interval_coeff: float = 0.12, default_interval: float = 0.01):
    lower_limit_0, upper_limit_0 = calculate_limit(
        target[0], interval_coeff, default_interval)
    lower_limit_1, upper_limit_1 = calculate_limit(
        target[1], interval_coeff, default_interval)
    lower_limit_2, upper_limit_2 = calculate_limit(
        target[2], interval_coeff, default_interval)

    # print(lower_limit_0, upper_limit_0)
    # print(lower_limit_1, upper_limit_1)
    # print(lower_limit_2, upper_limit_2)
    # print(candidate)

    if lower_limit_0 < candidate[0] < upper_limit_0:
        if lower_limit_1 < candidate[1] < upper_limit_1:
            if lower_limit_2 < candidate[2] < upper_limit_2:
                return True
    return False


def write_valid(path: str):
    file_name = "task.valid"
    if os.path.exists(path):
        if os.path.isdir(path):
            file_fullpath = os.path.join(path, file_name)
            with open(file_fullpath, 'w') as fs:
                fs.write("valid")


def compare_bounding_box(resize_fullpath: str, full_fullpath: str = "", new_full_fullpath: str = "",
                         render_daz: bool = False,
                         interval_coeff: float = 0.12, default_interval: float = 0.01):
    resize_aa, resize_bb = calculate_bounding_box(resize_fullpath)
    if len(full_fullpath) > 1:
        full_aa, full_bb = calculate_bounding_box(full_fullpath)
    if len(new_full_fullpath) > 1:
        new_full_aa, new_full_bb = calculate_bounding_box(new_full_fullpath)

    if render_daz:
        if len(full_fullpath) > 1:
            full_aa[[1, 2]] = full_aa[[2, 1]]
            full_bb[[1, 2]] = full_bb[[2, 1]]
        if len(new_full_fullpath) > 1:
            new_full_aa[[1, 2]] = new_full_aa[[2, 1]]
            new_full_bb[[1, 2]] = new_full_bb[[2, 1]]

    # print(resize_aa, full_aa, new_full_aa)
    # print(resize_bb, full_bb, new_full_bb)
    full_result = False
    new_full_result = False
    if len(full_fullpath) > 1:
        full_result = compare_aabb(full_aa, resize_aa, interval_coeff, default_interval) and compare_aabb(
            full_bb, resize_bb, interval_coeff, default_interval)
    if len(new_full_fullpath) > 1:
        new_full_result = compare_aabb(new_full_aa, resize_aa, interval_coeff, default_interval) and compare_aabb(
            new_full_bb, resize_bb, interval_coeff, default_interval)

    print(full_result, new_full_result)
    return full_result and new_full_result


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Verify obj start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mesh_path", type=str, default="",
                        help="path to obj")
    parser.add_argument('--render_daz', action='store_true',
                        help='have changed -ZY axis to YZ axis during blender process')
    parser.add_argument('--interval_coeff', type=float, default=0.12,
                        help='tolerance (ratio/scale) of the difference of two object\'s AABB')
    parser.add_argument('--default_interval', type=float, default=0.01,
                        help='minimal tolerance number of the difference of two object\'s AABB')
    args = parser.parse_args()

    mesh_path = args.mesh_path
    mesh_folder = os.path.split(mesh_path)[0]
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]
    mesh_extension = os.path.splitext(mesh_filename)[1]

    if "manifold_full" in mesh_path:
        resize_mesh_filename = mesh_filename.replace("manifold_full", "resize")
        resize_fullpath = os.path.join(mesh_folder, resize_mesh_filename)
        if not os.path.exists(resize_fullpath):
            resize_mesh_filename = mesh_filename.replace("_manifold_full", "")
        new_full_mesh_filename = mesh_filename.replace(
            "manifold_full", "manifold_new_full")
        full_mesh_filename = mesh_filename
    elif "manifold_new_full" in mesh_path:
        resize_mesh_filename = mesh_filename.replace(
            "manifold_new_full", "resize")
        resize_fullpath = os.path.join(mesh_folder, resize_mesh_filename)
        if not os.path.exists(resize_fullpath):
            resize_mesh_filename = mesh_filename.replace(
                "_manifold_new_full", "")
        full_mesh_filename = mesh_filename.replace(
            "manifold_new_full", "manifold_full")
        new_full_mesh_filename = mesh_filename

    resize_fullpath = os.path.join(mesh_folder, resize_mesh_filename)
    full_fullpath = os.path.join(mesh_folder, full_mesh_filename)
    new_full_fullpath = os.path.join(mesh_folder, new_full_mesh_filename)

    if compare_bounding_box(resize_fullpath, full_fullpath, new_full_fullpath, render_daz=args.render_daz,
                            interval_coeff=args.interval_coeff, default_interval=args.default_interval):
        print("Mesh at %s is valid..." % (mesh_path))
        write_valid(mesh_folder)
    else:
        print("Mesh at %s is not valid..." % (mesh_path))
