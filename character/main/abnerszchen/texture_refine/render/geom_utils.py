import os
import torch
import numpy as np
import trimesh
import math
import time
import numpy as np
import pymeshlab as pml

def calculate_scale_matrix(mesh_verts: np.array, standard_height: float = 1.98):
    import miniball
    original_mesh = trimesh.base.Trimesh(vertices=mesh_verts)
    hull_vertices = original_mesh.convex_hull.vertices
    bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(
        hull_vertices)

    obj_center = bounding_sphere_C
    length = 2*math.sqrt(bounding_sphere_r2)
    scale = standard_height / length
    translation = -1 * obj_center * scale
    transformation = np.array(
        [[scale, 0, 0, translation[0]],
         [0, scale, 0, translation[1]],
         [0, 0, scale, translation[2]],
         [0, 0, 0, 1]]
    )
    # print('debug bounding_sphere_C ', bounding_sphere_C)
    # print('debug bounding_sphere_r2 ', bounding_sphere_r2)
    # print('debug mesh_verts', np.min(mesh_verts), np.max(mesh_verts))
    return transformation


def mesh_normalized(imesh, max_len=1.98):
    """srender nromalized and move to center

    Args:
        imesh: Mesh
        max_len: _description_. Defaults to 1.98.
    """
    pos = imesh.v_pos
    # numpy [4, 4]
    transformation = calculate_scale_matrix(pos.detach().cpu().numpy())
    transformation = torch.tensor(transformation, dtype=torch.float32, device=pos.device)

    # pw, [N, 4]
    raw_points = torch.cat([pos, torch.ones_like(pos[..., 0:1])], dim=-1)

    # pc = Tcw * pw [N, 3]
    new_points = torch.bmm(transformation.unsqueeze(0), raw_points.permute(
        1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)[..., :3]

    imesh.v_pos = new_points.contiguous()

    # tf_new = calculate_scale_matrix(new_points.detach().cpu().numpy(), max_len)

    return transformation

def mesh_normalized_by_txt(imesh, transformation_txt):
    """srender nromalized and move to center

    Args:
        imesh: Mesh
        transformation_txt in config.json
    """
    pos = imesh.v_pos
    transformation = np.loadtxt(transformation_txt)
    transformation = torch.tensor(transformation, dtype=torch.float32, device=pos.device)
    
    # pw, [N, 4]
    raw_points = torch.cat([pos, torch.ones_like(pos[..., 0:1])], dim=-1)

    # pc = Tcw * pw [N, 3]
    new_points = torch.bmm(transformation.unsqueeze(0), raw_points.permute(
        1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)[..., :3]

    imesh.v_pos = new_points.contiguous()

    # tf_new = calculate_scale_matrix(new_points.detach().cpu().numpy(), max_len)

    return transformation

def try_mesh_normalized(imesh, attempts=5, sleep_time=1):
    for attempt in range(attempts):
        try:
            transformation = mesh_normalized(imesh)
            return transformation
        except Exception as e:
            if attempt < attempts - 1:
                time.sleep(sleep_time)
            else:
                raise e 

def poisson_mesh_reconstruction(points, normals=None):
    # points/normals: [N, 3] np.ndarray

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # outlier removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

    # normals
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals[ind])

    # visualize
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize
    o3d.visualization.draw_geometries([mesh])

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print(
        f"[INFO] poisson mesh reconstruction: {points.shape} --> {vertices.shape} / {triangles.shape}"
    )

    return vertices, triangles


def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_d)
        )
            # mincomponentdiag=pml.Percentage(min_d)    # for 2022.2

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.PureValue(remesh_size)
        )
        
            # iterations=3, targetlen=pml.Percentage(1)   # for 2022.2

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces



def clean_decimate_mesh(imesh, decimate_target=30000):
    vertices, triangles = imesh.v_pos.cpu().numpy().astype(np.float32), imesh.t_pos_idx.cpu().numpy().astype(np.int32)
    vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)
    
    imesh.v_pos = torch.from_numpy(vertices.astype(np.float32)).contiguous().to(imesh.v_pos.device)
    imesh.t_pos_idx = torch.from_numpy(triangles.astype(np.int32)).contiguous().to(imesh.t_pos_idx.device)
    return imesh

def get_min_max(point_cloud):
    min_values, _ = torch.min(point_cloud, dim=0)
    max_values, _ = torch.max(point_cloud, dim=0)

    x_min, y_min, z_min = min_values
    x_max, y_max, z_max = max_values

    return x_min, x_max, y_min, y_max, z_min, z_max

def get_xyz_range(pos):
    x_min, x_max, y_min, y_max, z_min, z_max = get_min_max(pos)
    
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)
    z_range = abs(z_max - z_min)    
    return x_range, y_range, z_range

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0],
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def align_z_mesh(imesh, thr_range=1.8):
    pos = imesh.v_pos   # N,3
    x_range, y_range, z_range = get_xyz_range(pos)
    print(f'debug raw : y_range {y_range} z_range {z_range}')
    
    if y_range >= thr_range:    # bade case
        print('[AILGN] need rot align_z_mesh !')
        Rot_x = rotate_x(-np.pi / 2, pos.device)[:3, :3]
        
        # p' = R * p [3, 3] * [N, 3]
        # new_points = torch.bmm(Rot_x.unsqueeze(0), pos.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        new_points = torch.matmul(pos, Rot_x.T)
        imesh.v_pos = new_points.contiguous()
            
    x_range, y_range, z_range = get_xyz_range(imesh.v_pos)
    if z_range < thr_range:
        print(f'[Warn] need fixrot align_z_mesh : y_range {y_range} z_range {z_range}')
    return imesh


def align_y_mesh(imesh, thr_range=1.8):
    """make mesh alway y up

    Args:
        imesh: _description_
        thr_range: _description_. Defaults to 1.5.

    Returns:
        _description_
    """
    pos = imesh.v_pos   # N,3
    x_range, y_range, z_range = get_xyz_range(pos)
    print(f'debug raw : y_range {y_range} z_range {z_range}')
    
    if z_range >= thr_range:    # bade case
        print('[AILGN] need rot align_y_mesh!')
        Rot_x = rotate_x(np.pi / 2, pos.device)[:3, :3]
        
        # p' = R * p [3, 3] * [N, 3]
        # new_points = torch.bmm(Rot_x.unsqueeze(0), pos.permute(1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        new_points = torch.matmul(pos, Rot_x.T)
        imesh.v_pos = new_points.contiguous()
            
    x_range, y_range, z_range = get_xyz_range(imesh.v_pos)
    if y_range < thr_range:
        print(f'[Warn] need fixrot align_y_mesh : y_range {y_range} z_range {z_range}')
    return imesh

