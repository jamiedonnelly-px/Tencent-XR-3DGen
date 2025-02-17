import argparse
import os
import time

import numpy as np
import trimesh


def sample_surface_tex(mesh, num, weight):
    v_points, v_tri, v_rgba = trimesh.sample.sample_surface(
        mesh, num, weight, sample_color=True)
    if v_rgba is None:
        v_rgba = mesh.visual.material.main_color * np.ones([num, 4])
    v_rgba = v_rgba / 255.
    v_colors = v_rgba[:, :3] * v_rgba[:, 3:]
    v_normals = mesh.face_normals[v_tri]
    return v_points, v_normals, v_colors


def sample_points_tex(meshs, sample_number: int = 500000):
    areas = [m.area for m in meshs]
    tris = [len(m.triangles) for m in meshs]

    uniform_sample_number = [
        int(sample_number // 2 * a / np.sum(areas)) for a in areas]
    trianlge_sample_number = [
        int(sample_number // 2 * t / np.sum(tris)) for t in tris]
    uniform_sample_number[np.argmin(
        uniform_sample_number)] += sample_number // 2 - np.sum(uniform_sample_number)
    trianlge_sample_number[np.argmin(
        trianlge_sample_number)] += sample_number // 2 - np.sum(trianlge_sample_number)

    all_points = []
    all_normals = []
    all_colors = []

    for i, mesh in enumerate(meshs):
        uni_points, uni_normals, uni_colors = sample_surface_tex(
            mesh, uniform_sample_number[i], None)
        tri_points, tri_normals, tri_colors = sample_surface_tex(
            mesh, trianlge_sample_number[i], np.ones(tris[i]) / tris[i])
        all_points.append(np.concatenate([uni_points, tri_points]))  # [N,3]
        all_colors.append(np.concatenate([uni_colors, tri_colors]))  # [N,3]
        all_normals.append(np.concatenate([uni_normals, tri_normals]))

    all_points = np.concatenate(all_points)
    all_colors = np.concatenate(all_colors)
    all_normals = np.concatenate(all_normals)
    assert len(all_colors) == len(all_points) == len(all_normals)
    return all_points, all_colors, all_normals


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Texture sample starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mesh_path", type=str, default="",
                        help="path to manifold obj")
    parser.add_argument("--output_folder", type=str, default="",
                        help="output folder of sampled points")
    parser.add_argument("--transform_path", type=str, default="",
                        help="input transformation txt path")
    parser.add_argument("--texture_sample_number", type=int, default=500000,
                        help="number of sdf sampled points on surface")
    args = parser.parse_args()

    mesh_path = args.mesh_path
    output_folder = args.output_folder
    transform_path = args.transform_path
    texture_sample_number = args.texture_sample_number

    internal_rotation = trimesh.transformations.euler_matrix(np.pi / 2, 0.0, 0.0, 'rxyz')

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    texture_sample_folder = os.path.join(output_folder, "texture")
    if not os.path.exists(texture_sample_folder):
        os.mkdir(texture_sample_folder)

    imported_mesh = trimesh.load(mesh_path)

    # if z_up:
    #     imported_mesh.apply_transform(internal_rotation)

    if isinstance(imported_mesh, trimesh.scene.scene.Scene):
        imported_mesh_list = imported_mesh.dump(False)
    elif isinstance(imported_mesh, trimesh.base.Trimesh):
        imported_mesh_list = np.array([imported_mesh])
    else:
        raise NotImplementedError

    ready_for_sampling_mesh_list = []
    for mesh in imported_mesh_list:
        has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
        has_vertex_color = hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None
        has_img = hasattr(mesh.visual.material, "image") and mesh.visual.material.image is not None

        if (has_uv and has_img):
            ready_for_sampling_mesh_list.append(mesh)
        elif (has_uv and not has_img):
            ready_for_sampling_mesh_list.append(mesh)
        elif (not has_uv and not has_img and has_vertex_color):
            ready_for_sampling_mesh_list.append(mesh)
        # else:
        #     print("Cannot find uv of one part of mesh...")
        #     exit(-1)

    if sum([len(m.triangles) for m in ready_for_sampling_mesh_list]) < 100:
        print("Mesh triangle number is smaller than 100...")
        exit(-1)

    T = np.loadtxt(transform_path)
    for mesh in ready_for_sampling_mesh_list:
        mesh.apply_transform(T)

    sample_results = sample_points_tex(ready_for_sampling_mesh_list, texture_sample_number)
    sampled_points = sample_results[0].astype(np.float32)
    sampled_colors = sample_results[1].astype(np.float32)
    sampled_normals = sample_results[2].astype(np.float32)

    texture_points_npy = os.path.join(texture_sample_folder, ("texture_point_%i.npy" % texture_sample_number))
    texture_colors_npy = os.path.join(texture_sample_folder, ("texture_colors_%i.npy" % texture_sample_number))
    texture_normals_npy = os.path.join(texture_sample_folder, ("texture_normals_%i.npy" % texture_sample_number))
    np.save(texture_points_npy, sampled_points)
    np.save(texture_colors_npy, sampled_colors)
    np.save(texture_normals_npy, sampled_normals)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Texture sample finished. Local time is %s" % (local_time_str))
