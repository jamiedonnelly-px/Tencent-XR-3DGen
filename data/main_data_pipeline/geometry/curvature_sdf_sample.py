import argparse
import inspect
import math
import os
import time
from functools import wraps

import h5py
import miniball
import numpy as np
import scipy
import trimesh


def euler_to_rotation_matrix(euler_angles):
    r = scipy.spatial.transform.Rotation.from_euler('xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def logtime(print_time=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_info = inspect.getfile(func)
            func_lineno = inspect.findsource(func)[1]
            start_time = time.time()
            ret = func(*args, **kwargs)
            if print_time:
                print(f"{func_info}({func_lineno})<{func.__name__}>() {(time.time() - start_time) * 1e3} ms")
            return ret

        return wrapper

    return decorator


def sample_barycentric(N):
    '''
    uniformly sample triangular barycentric coordinates

    Inputs:
        - N: integer, how many sample to return

    Returns:
        - bary: barycentric coordinates of shape [N,3]
    '''
    uv = np.random.rand(N, 2)
    reflect_mask = (uv.sum(1) > 1)
    uv[reflect_mask] = 1 - uv[reflect_mask]  # reflect points across the line u+v=1

    w = 1 - np.sum(uv, axis=1, keepdims=True)
    bary = np.concatenate((uv, w), axis=1)  # [N,3]
    return bary


def sample_space(N, space="cube"):
    '''
    uniformly sample points inside cube [-1,1]^3 or unit sphere

    Inputs:
        - N: integer, how many sample to return
        - space: str "cube" or "sphere"

    Returns:
        - points: of shape [N,3]
    '''
    assert space in ["cube", "sphere"]
    if space == "cube":
        points = np.random.rand(N, 3) * 2 - 1
    elif space == "sphere":
        points = np.random.randn(N, 3)
        radii = np.random.rand(N, 1)
        points = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-10) * radii
    return points


def compute_barycentric(triangles, queries):
    '''
    solve for barycentric coordinates of queries in the correponding triangles.
    it is assumed queries are indeed inside triangles, otherwise behaviour is undetermined.

    Inputs:
        - queries: float array of shape [N,3], query points
        - triangles: float array of shape [N, 3=verts, 3=xyz]
    Returns:
        - bary: barycentric coordinates of shape [N,3]
    '''

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]

    e1 = v1 - v0
    e2 = v2 - v0
    eq = queries - v0

    d00 = (e1 * e1).sum(1)
    d01 = (e1 * e2).sum(1)
    d11 = (e2 * e2).sum(1)
    d20 = (eq * e1).sum(1)
    d21 = (eq * e2).sum(1)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    bary = np.stack((u, v, w), axis=1)
    bary[denom < 1e-10] = (3 ** -1)

    return bary


def compute_closest_point(triangles, queries):
    '''
    compute the closest point on each triangle to corresponding query points.

    Inputs:
        - triangles: float array of shape [N, 3=verts, 3=xyz], the vertices of triangles
        - queries: float array of shape [N, 3], the 3D query points

    Returns:
        - closest: float array of shape [N, 3], the closest points on each triangle
        - bary: float array of shape [N, 3], the bary coordinates of closest
    '''

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]

    e1 = v1 - v0
    e2 = v2 - v0
    eq = queries - v0

    d00 = (e1 * e1).sum(1)
    d01 = (e1 * e2).sum(1)
    d11 = (e2 * e2).sum(1)
    d20 = (eq * e1).sum(1)
    d21 = (eq * e2).sum(1)

    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    closest = v0 + np.expand_dims(u, axis=1) * e1 + np.expand_dims(v, axis=1) * e2

    # out side masks
    mask_u = u < 0
    mask_v = v < 0
    mask_sum = u + v > 1

    # if u < 0, project onto the edge v0-v2
    closest[mask_u] = v0[mask_u] + np.clip(
        np.sum((queries[mask_u] - v0[mask_u]) * e2[mask_u], axis=1) /
        np.sum(e2[mask_u] * e2[mask_u], axis=1),
        0, 1).reshape(-1, 1) * e2[mask_u]

    v[mask_u] = np.clip(np.sum((queries[mask_u] - v0[mask_u]) * e2[mask_u], axis=1) /
                        np.sum(e2[mask_u] * e2[mask_u], axis=1), 0, 1)
    u[mask_u] = 0
    w[mask_u] = 1 - v[mask_u]

    # if v < 0, project onto the edge v0-v1
    closest[mask_v] = v0[mask_v] + np.clip(
        np.sum((queries[mask_v] - v0[mask_v]) * e1[mask_v], axis=1) /
        np.sum(e1[mask_v] * e1[mask_v], axis=1),
        0, 1).reshape(-1, 1) * e1[mask_v]

    u[mask_v] = np.clip(np.sum((queries[mask_v] - v0[mask_v]) * e1[mask_v], axis=1) /
                        np.sum(e1[mask_v] * e1[mask_v], axis=1), 0, 1)
    v[mask_v] = 0
    w[mask_v] = 1 - u[mask_v]

    # if u + v > 1, project onto the edge v1-v2
    closest[mask_sum] = v1[mask_sum] + np.clip(
        np.sum((queries[mask_sum] - v1[mask_sum]) * (v2 - v1)[mask_sum], axis=1) /
        np.sum((v2 - v1)[mask_sum] * (v2 - v1)[mask_sum], axis=1),
        0, 1).reshape(-1, 1) * (v2 - v1)[mask_sum]

    u[mask_sum] = np.clip(np.sum((queries[mask_sum] - v1[mask_sum]) * (v2 - v1)[mask_sum], axis=1) /
                          np.sum((v2 - v1)[mask_sum] * (v2 - v1)[mask_sum], axis=1), 0, 1)
    v[mask_sum] = 1 - u[mask_sum]
    w[mask_sum] = 0

    bary = np.stack((u, v, w), axis=1)
    bary[denom < 1e-10] = (3 ** -1)

    return closest, bary


def compute_face_area(verts, faces):
    '''
    compute triangle face area

    Inputs:
        - verts: float array of shape [n_verts, 3]
        - faces: int array of shape [n_faces, 3], values in [0, n_verts)

    Returns:
        - area: 1d float array of shape [n_faces]
    '''
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e0 = v1 - v0
    e2 = v0 - v2
    cross_product = np.cross(e0, e2)
    area = 0.5 * np.linalg.norm(cross_product, axis=1)
    return area


def compute_angles(verts, faces):
    '''
    compute triangle face angles

    Inputs:
        - verts: float array of shape [n_verts, 3]
        - faces: int array of shape [n_faces, 3], values in [0, n_verts)

    Returns:
        - angles: 2d float array of shape [n_faces, 3]
    '''
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # edges, [n_faces, 3]
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # edge lengths, [n_faces, 1]
    len0 = np.linalg.norm(e0, axis=1, keepdims=True)
    len1 = np.linalg.norm(e1, axis=1, keepdims=True)
    len2 = np.linalg.norm(e2, axis=1, keepdims=True)

    # normalised edges, [n_faces, 3]
    norm_e0 = e0 / len0
    norm_e1 = e1 / len1
    norm_e2 = e2 / len2

    # angles, [n_faces]
    angle0 = np.arccos(np.clip(np.sum(-norm_e2 * norm_e0, axis=1), -1.0, 1.0))
    angle1 = np.arccos(np.clip(np.sum(-norm_e0 * norm_e1, axis=1), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.sum(-norm_e1 * norm_e2, axis=1), -1.0, 1.0))

    return np.stack([angle0, angle1, angle2], axis=1)


def compute_cell_area(verts, faces, cell_type="barycentric", epsilon=1e-10):
    '''
    Compute the cell area for each face surrounding each vertex

    Inputs:
        - verts: float array of shape [n_verts, 3], vertex coordinates
        - faces: int array of shape [n_faces, 3], vertex indices for each face
        - cell_type: a string for cell type, can be "barycentric", "voronoi", "mixed_voronoi", or "mixed_voronoi_scaled"
            - barycentric: cells are splitted by barycentre
            - voronoi: cells are splitted by circumcentre, areas could be negative for obtuse triangles
            - mixed_voronoi: vonoroi for acute triangles and barycentric for obtuse ones
            - mixed_voronoi_scaled: voronoi for acute triangles, and for obtuse triangles, 1.5x bary area if the angle at vertex is obtuse or 0.75x bary area otherwise

    Returns:
        - areas: float array of shape [n_faces, 3], cell areas of each face's vertex
    '''

    #        v0_
    #      /  |\
    #    e0     e2
    #   |/_       \
    #   v1 - e1 -> v2

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # edges, [n_faces, 3]
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # edge lengths, [n_faces, 1]
    len0 = np.linalg.norm(e0, axis=1, keepdims=True)
    len1 = np.linalg.norm(e1, axis=1, keepdims=True)
    len2 = np.linalg.norm(e2, axis=1, keepdims=True)

    # normalised edges, [n_faces, 3]
    norm_e0 = e0 / len0
    norm_e1 = e1 / len1
    norm_e2 = e2 / len2

    # angles, [n_faces]
    angle0 = np.arccos(np.clip(np.sum(-norm_e2 * norm_e0, axis=1), -1.0, 1.0))
    angle1 = np.arccos(np.clip(np.sum(-norm_e0 * norm_e1, axis=1), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.sum(-norm_e1 * norm_e2, axis=1), -1.0, 1.0))

    # circumcentric triangle areas, [n_faces]
    c0 = len1.flatten() ** 2 / (4 * np.tan(angle0) + np.where(angle0 < np.pi / 2, epsilon, -epsilon))
    c1 = len2.flatten() ** 2 / (4 * np.tan(angle1) + np.where(angle1 < np.pi / 2, epsilon, -epsilon))
    c2 = len0.flatten() ** 2 / (4 * np.tan(angle2) + np.where(angle2 < np.pi / 2, epsilon, -epsilon))

    # cell areas, [n_faces, 3]
    bary_area = np.repeat(((c0 + c1 + c2) / 3)[:, np.newaxis], 3, axis=1)
    voronoi_area = np.stack(((c1 + c2) / 2, (c0 + c2) / 2, (c0 + c1) / 2), axis=1)

    obtuse_angle_mask = np.stack((angle0, angle1, angle2), axis=1) > (np.pi / 2)  # [n_faces, 3]
    obtuse_triangle_mask = np.max(obtuse_angle_mask, axis=1, keepdims=True)  # [n_faces, 1]

    if cell_type == "barycentric":
        return bary_area
    elif cell_type == "voronoi":
        return voronoi_area
    elif cell_type == "mixed_voronoi":
        return np.where(obtuse_triangle_mask, bary_area, voronoi_area)
    elif cell_type == "mixed_voronoi_scaled":
        return np.where(obtuse_triangle_mask, np.where(obtuse_angle_mask, bary_area * 1.5, bary_area * 0.75),
                        voronoi_area)
    else:
        raise ValueError(f"unrecognised cell type \"{cell_type}\"")


def sample_surface_points(verts, faces, N):
    '''
    sample surface points uniformly ny surface area

    Inputs:
        - verts: float array of shape [n_verts, 3]
        - faces: int array of shape [n_faces, 3], values in [0, n_verts)
        - N: int, number of samples

    Returns:
        - points: float array of shape [N, 3], points' 3D coordinates
        - face_id: int array of shape [N] indicating the faces each point belongs to, values in [0, n_faces)
        - bary: float array of shape [N,3], barycentric coordinates of points inside each face, each row sums to 1
    '''
    areas = compute_face_area(verts, faces)
    face_id = np.random.choice(len(faces), size=N, p=areas / np.sum(areas))  # [N]

    bary = sample_barycentric(N)  # [N,3]
    points = (verts[faces[face_id]] * np.expand_dims(bary, axis=2)).sum(axis=1)  # [N,3]
    return points, face_id, bary


def compute_vert_normals(verts, faces, weighting='area'):
    '''
    verts: numpy array [n_verts, 3], float
    faces: numpy array [n_faces, 3], int in [0, n_verts)
    weighting: 'area' or 'angle', defines the weighting strategy
    returns: numpy array [n_verts, 3], float vert normals

    Interpolation is weighted either by surrounding face area or by angle.
    '''

    assert weighting in ['area', 'angle']

    i0, i1, i2 = faces[:, 0].astype(int), faces[:, 1].astype(int), faces[:, 2].astype(int)
    v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :]

    face_normals = np.cross(v1 - v0, v2 - v0, axis=-1)  # [n_faces, 3]

    # Compute per-face areas (used in both strategies)
    face_areas = np.linalg.norm(face_normals, axis=-1, keepdims=True) * 0.5  # [n_faces, 1]

    if weighting == 'area':
        weights = face_areas  # [n_faces, 1]
    elif weighting == 'angle':
        def compute_angles(a, b, c):
            ab = b - a
            ac = c - a
            ab_norm = np.linalg.norm(ab, axis=-1)
            ac_norm = np.linalg.norm(ac, axis=-1)
            cos_angle = np.sum(ab * ac, axis=-1) / (ab_norm * ac_norm + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.arccos(cos_angle)

        angle0 = compute_angles(v1, v2, v0)
        angle1 = compute_angles(v2, v0, v1)
        angle2 = compute_angles(v0, v1, v2)

        weights = np.stack([angle0, angle1, angle2], axis=1)  # [n_faces, 3]

    vn = np.zeros_like(verts)  # [n_verts, 3]

    if weighting == 'area':
        contrib = face_normals * weights  # [n_faces, 3]
        np.add.at(vn, i0, contrib)
        np.add.at(vn, i1, contrib)
        np.add.at(vn, i2, contrib)
    elif weighting == 'angle':
        contrib0 = face_normals * weights[:, 0:1]  # [n_faces, 3]
        contrib1 = face_normals * weights[:, 1:2]
        contrib2 = face_normals * weights[:, 2:3]

        np.add.at(vn, i0, contrib0)
        np.add.at(vn, i1, contrib1)
        np.add.at(vn, i2, contrib2)

    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    valid = norms > 1e-20
    vn_normalized = np.zeros_like(vn)

    vn_normalized[valid[:, 0]] = vn[valid[:, 0]] / norms[valid[:, 0]]
    vn_normalized[~valid[:, 0]] = np.array([0.0, 0.0, 1.0])  # default normal

    return vn_normalized


def compute_vert_curvatures(verts, faces, cell_type="mixed_voronoi_scaled"):
    '''
    compute per-vertex mean and gaussian curvatures

    Inputs:
        - verts: float array of shape [n_verts, 3], vertex coordinates
        - faces: int array of shape [n_faces, 3], each row representing indices of vertices that form a triangle

    Returns:
        - mean_curv: float array of shape [n_verts], per-vertex mean curvatures
        - gauss_curv: float array of shape [n_verts], per-vertex Gaussian curvatures
    '''
    n_verts = len(verts)
    n_faces = len(faces)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    face_normals = np.cross(e0, -e2)
    face_areas = 0.5 * np.linalg.norm(face_normals, axis=1)

    angles = compute_angles(verts, faces)  # [n_faces, 3]
    face_cell_areas = compute_cell_area(verts, faces, cell_type=cell_type)  # [n_faces, 3]

    angle_deficits = np.zeros(n_verts) + 2 * np.pi
    vert_areas = np.zeros(n_verts)

    np.add.at(angle_deficits, faces.flatten(), -angles.flatten())
    np.add.at(vert_areas, faces.flatten(), face_cell_areas.flatten())

    gauss_curv = angle_deficits / (vert_areas + 1e-10)

    mean_curv_vector = np.zeros((n_verts, 3))
    cot_angles = np.zeros((n_faces, 3))

    for i in range(3):
        va = verts[faces[:, (i + 1) % 3]] - verts[faces[:, i]]
        vb = verts[faces[:, (i + 2) % 3]] - verts[faces[:, i]]

        cross = np.cross(va, vb)
        sin_theta = np.linalg.norm(cross, axis=1) + 1e-10
        cos_theta = np.sum(va * vb, axis=1)

        cot_theta = cos_theta / sin_theta
        cot_angles[:, i] = cot_theta

    mean_curv_vector = np.zeros((n_verts, 3))

    # build laplace-beltrami operator per edge
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                        faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                        faces[:, 2], faces[:, 0], faces[:, 1]])
    W = 0.5 * np.concatenate([cot_angles[:, 2], cot_angles[:, 0], cot_angles[:, 1],
                              cot_angles[:, 1], cot_angles[:, 2], cot_angles[:, 0]])

    delta = W[:, np.newaxis] * (verts[J] - verts[I])  # [n_edges=6*n_faces, 3]
    np.add.at(mean_curv_vector, I, delta)
    mean_curv = 0.5 * np.linalg.norm(mean_curv_vector, axis=1) / (vert_areas + 1e-10)

    return mean_curv, gauss_curv


def compute_principal_curvatures(mean_curv, gauss_curv):
    '''
    compute principal curvatures given mean and gauss curvatures.
    principal curvatures defines the maximal and minimal curvatures in two orthogonal directions tangent to surface
    e.g. a cylindrical surface in any rotation has principal curvatures k1=1/radius and k2=0,
         and a spherical surface has k1=k2=1/radius

    Inputs:
        - mean_curv, gauss_curv: both of same shape [...]
    Returns:
        - principal_curv_major: maximal curvature k1 of same shape as inputs
        - principal_curv_minor: minimal curvature k2 of same shape as inputs

    '''

    discriminant = mean_curv ** 2 - gauss_curv
    discriminant = np.maximum(discriminant, 0.0)
    sqrt_discriminant = np.sqrt(discriminant)

    principal_curv_major = mean_curv + sqrt_discriminant
    principal_curv_minor = mean_curv - sqrt_discriminant

    return principal_curv_major, principal_curv_minor


def barycentric_interpolate(faces, vert_features, face_id, bary):
    '''

    Inputs:
        - faces: [F,3]
        - vert_features: [V, c]
        - face_id: [N]
        - bary: [N,3]

    Returns:
        - features: [N,c]
    '''

    face_verts_features = vert_features[faces[face_id]]  # [N,3,c]
    return (bary[..., None] * face_verts_features).sum(1)


@logtime(print_time=True)
def find_nearest_surface(verts, faces, queries, o3d_scene=None, sdf_scene=None, backend="o3d"):
    '''

    Inputs:
        - verts: [V,3]
        - faces: [F,3]
        - queries: [N,3]
        - backend: "o3d" slow but exact, "sdf" fast but approximate
    Returns:
        - closest: float array of shape [N, 3], closest surface points' 3D coordinates
        - face_id: int array of shape [N] indicating the faces each point belongs to, values in [0, n_faces)
        - bary: float array of shape [N,3], barycentric coordinates of closest inside each face, each row sums to 1
        - distances: float array of shape [N], unsigned distances
        - inside: boolean of shape [N], whether queries are inside mesh (reliable with watertight mesh)
    '''

    assert backend in ["o3d", "sdf"]

    import pysdf

    if sdf_scene is None:
        sdf_scene = pysdf.SDF(verts, faces)
    inside = sdf_scene.contains(queries)

    if backend == "o3d":
        import open3d as o3d

        if o3d_scene is None:
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
            o3d_scene = o3d.t.geometry.RaycastingScene()
            triangle_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            o3d_scene.add_triangles(triangle_mesh)

        query_tensor = o3d.core.Tensor(queries.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        results = o3d_scene.compute_closest_points(query_tensor)

        closest = results['points'].numpy()
        face_id = results['primitive_ids'].numpy()
        uv = results['primitive_uvs'].numpy()
        distances = np.linalg.norm(closest - queries, axis=1, ord=2)
        bary = np.concatenate((1 - uv.sum(1, keepdims=True), uv), axis=-1)

    else:

        face_id = sdf_scene.nearest_triangle(queries)
        closest, bary = compute_closest_point(verts[faces[face_id]], queries)
        distances = np.linalg.norm(closest - queries, axis=1, ord=2)

    return closest, face_id, bary, distances, inside, o3d_scene, sdf_scene


class CurvatureSampler:

    def __init__(self, verts, faces):

        self.verts = verts
        self.faces = faces
        self.vert_normals = compute_vert_normals(verts, faces, 'area')
        self.vert_mean_cur, self.vert_gauss_cur = compute_vert_curvatures(verts, faces, "mixed_voronoi_scaled")

        self.vert_norm_mc_gc = np.concatenate(
            (self.vert_normals, self.vert_mean_cur[:, None], self.vert_gauss_cur[:, None]), axis=1)  # [V,5]
        self.o3d_scene, self.sdf_scene = None, None

    def sample_surface(self, N):
        '''
        uniformly sample surface and return surface normal and mean and gauss curvatures

        returns:
            - points of shape [N,3]
            - face_id of shape [N] values in [0,F), indices of face that each sample lies in
            - bary of shape [N,3] in [0,1], barycentric coordinate of sample points inside each face
            - normal of shape [N,3]
            - mean_cur, gauss_cur of shape [N,1]
        '''

        points, face_id, bary = sample_surface_points(self.verts, self.faces, N)
        norm_mc_gc = barycentric_interpolate(self.faces, self.vert_norm_mc_gc, face_id, bary)
        normal, mean_cur, gauss_cur = np.split(norm_mc_gc, (3, 4), axis=1)

        surface_data = {}
        surface_data["surface_points"] = points
        surface_data["surface_facess"] = face_id
        surface_data["surface_normal"] = normal
        surface_data["surface_barycenter"] = bary
        return surface_data

    def sample_near_surface(self, N, distribution="gauss", std=1e-2, compute_closest="o3d"):
        '''
        sample near surface points by first sampling on surface before randomly perturbing surface
        points along normal direction

        Inputs:
            - integer N number of samples
            - distribution: "uniform" or "gauss", distribution of distance away from surface
            - std: float, standard deviation of distance
            - compute_closest: if "o3d" or "sdf", returned attributes are computed from the actual nearest point on surface
                     using specified method. "o3d" is slow but exact, "sdf" is much faster but may give non-accurate results
                     when some faces are large. if None, closest points are assumed to be the on-surface point that were
                     sampled before random pertubations are applied.

        Returns:
            - points of shape [N,3]
            - closest: closest surface point of shape [N,3], depending on whether or not exact, this can either
                       be the actual closest surface point or the original surface sampling before pertubation
            - face_id of shape [N] values in [0,F), indices of face that the closest surface point lies in
            - bary of shape [N,3] in [0,1], barycentric coordinate of the closest surface point
            - normal of closest point of shape [N,3]
            - mean_cur, gauss_cur of closest points, of shape [N,1]
            - dist, to closest point, of shape [N,1] unsigned distance from surface
            - inside, boolean of shape [N], whether points are inside surface
        '''
        closest, face_id, bary, normal, mean_cur, gauss_cur = self.sample_surface(N)

        if distribution == "uniform":
            perturb = (np.random.rand(N, 1) * 2 - 1) * (3 ** 0.5 * std)  # [N,1]
        elif distribution == "gauss":
            perturb = np.random.randn(N, 1) * std  # [N,1]

        perturb[perturb == 0] = 1e-10
        points = perturb * normal + closest

        if compute_closest:
            # recompute closest point, dist, normal and curvatures
            closest, face_id, bary, dist, inside, self.o3d_scene, self.sdf_scene = find_nearest_surface(self.verts,
                                                                                                        self.faces,
                                                                                                        points,
                                                                                                        self.o3d_scene,
                                                                                                        self.sdf_scene,
                                                                                                        backend=compute_closest)
            norm_mc_gc = barycentric_interpolate(self.faces, self.vert_norm_mc_gc, face_id, bary)
            normal, mean_cur, gauss_cur = np.split(norm_mc_gc, (3, 4), axis=1)
        else:
            dist = np.abs(perturb)
            inside = (perturb > 0)

        near_surface_data = {}
        near_surface_data["near_surface_points"] = points
        near_surface_data["near_surface_faces"] = face_id
        near_surface_occupancy = np.expand_dims(inside, axis=-1)
        near_surface_data["near_surface_occupancy"] = near_surface_occupancy
        signed_distance = np.where(near_surface_occupancy, dist, -dist)
        near_surface_data["near_surface_sdf"] = signed_distance
        near_surface_data["near_surface_normal"] = normal
        near_surface_data["near_surface_barycenter"] = bary
        return near_surface_data

    def sample_space(self, N, space="cube", compute_closest="o3d"):
        '''
         uniformly sample points inside cube [-1,1]^3 or sphere

        Inputs:
            - N: integer, how many sample to return
            - space: str "cube" or "sphere"

        Returns:
            - points of shape [N,3]
            - closest: closest surface point of shape [N,3], depending on whether or not exact, this can either
                       be the actual closest surface point or the original surface sampling before pertubation
            - face_id of shape [N] values in [0,F), indices of face that the closest surface point lies in
            - bary of shape [N,3] in [0,1], barycentric coordinate of the closest surface point
            - normal of closest point of shape [N,3]
            - mean_cur, gauss_cur of closest points, of shape [N,1]
            - dist, to closest point, of shape [N,1] unsigned distance from surface
            - inside, boolean of shape [N], whether points are inside surface
        '''
        points = sample_space(N, space)
        closest, face_id, bary, dist, inside, self.o3d_scene, self.sdf_scene = find_nearest_surface(self.verts,
                                                                                                    self.faces, points,
                                                                                                    self.o3d_scene,
                                                                                                    self.sdf_scene,
                                                                                                    backend=compute_closest)
        norm_mc_gc = barycentric_interpolate(self.faces, self.vert_norm_mc_gc, face_id, bary)
        normal, mean_cur, gauss_cur = np.split(norm_mc_gc, (3, 4), axis=1)

        space_data = {}
        space_data["space_points"] = points
        space_data["space_faces"] = face_id
        space_occupancy = np.expand_dims(inside, axis=-1)
        space_data["space_occupancy"] = space_occupancy
        signed_distance = np.where(space_occupancy, dist, -dist)
        space_data["space_sdf"] = signed_distance
        space_data["space_normal"] = normal
        space_data["space_barycenter"] = bary
        return space_data


def generate_sdf(mesh: trimesh.Trimesh,
                 space_sample_number: int = 500000,
                 near_surface_sample_number: int = 500000,
                 surface_sample_number: int = 500000):
    mesh_sampler = CurvatureSampler(verts=mesh.vertices, faces=mesh.faces)

    t_current = time.time()
    local_time = time.localtime(t_current)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Start sample mesh in spaces at time %s ....." % local_time_str)

    space_data = mesh_sampler.sample_space(N=space_sample_number)

    # t_current = time.time()
    # local_time = time.localtime(t_current)
    # local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    # print("Start sample mesh near surface at time %s ....." % local_time_str)
    #
    # near_surface_data = mesh_sampler.sample_near_surface(N=near_surface_sample_number)
    #
    # t_current = time.time()
    # local_time = time.localtime(t_current)
    # local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    # print("Start sample mesh on surface at time %s ....." % local_time_str)
    #
    # surface_data = mesh_sampler.sample_near_surface(N=surface_sample_number)

    return space_data  # , near_surface_data, surface_data


def shuffle_numpy_array(data_length: int, data_struct: dict):
    indices = np.arange(data_length)
    np.random.shuffle(indices)
    result = {}
    for data_name in data_struct:
        data_shuffled = data_struct[data_name][indices]
        result[data_name] = data_shuffled
    return result


def save_struct_as_numpy(data_struct: dict, output_folder: str, data_length: int):
    for data_name in data_struct:
        npy_filename = os.path.join(output_folder, (data_name + ("_%i.npy" % data_length)))
        np.save(npy_filename, data_struct[data_name])


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Curvature based SDF calculation starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mesh_path", type=str, default="",
                        help="path to manifold obj")
    parser.add_argument("--output_folder", type=str, default="",
                        help="output folder of sampled points")
    parser.add_argument("--transform_path", type=str, default="",
                        help="input transformation txt path")
    parser.add_argument("--z_transform_path", type=str, default="",
                        help="input transformation txt path for z_up coordinate system")
    parser.add_argument("--standard_height", type=float, default=1.92,
                        help="standard height of the mesh")
    parser.add_argument("--space_sample_number", type=int, default=500000,
                        help="number of sdf sampled in all spaces")
    parser.add_argument("--near_surface_sample_number", type=int, default=500000,
                        help="number of sdf sampled points near surface")
    parser.add_argument("--surface_sample_number", type=int, default=500000,
                        help="number of sdf sampled points on surface")
    parser.add_argument('--sample_format', type=str, default="h5_chunk",
                        help='output format, choose between h5,h5_chunk,npy')
    parser.add_argument('--chunk_size', type=int, default=4096,
                        help='chunked storage size, only used when sample_format=h5_chunk')
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle output data sequence')
    args = parser.parse_args()

    mesh_path = args.mesh_path
    output_folder = args.output_folder
    transform_path = args.transform_path
    z_transform_path = args.z_transform_path
    standard_height = args.standard_height
    space_sample_number = args.space_sample_number
    near_surface_sample_number = args.near_surface_sample_number
    surface_sample_number = args.surface_sample_number
    sample_format = args.sample_format
    chunk_size = args.chunk_size
    shuffle = args.shuffle

    internal_rotation = euler_to_rotation_matrix(np.array([90, 0.0, 0.0]))
    inverse_internal_rotation = np.linalg.inv(internal_rotation)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    geometry_sample_folder = os.path.join(output_folder, "geometry")
    if not os.path.exists(geometry_sample_folder):
        os.mkdir(geometry_sample_folder)

    original_mesh = trimesh.load(mesh_path)
    hull_vertices = original_mesh.convex_hull.vertices

    try:
        bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
    except:
        time.sleep(0.1)
        print("Miniball failed. Retry once...............")
        try:
            bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
        except:
            time.sleep(0.1)
            print("Miniball failed. Retry second time...............")
            try:
                bounding_sphere_C, bounding_sphere_r2 = miniball.get_bounding_ball(hull_vertices)
            except:
                time.sleep(0.1)
    obj_center = bounding_sphere_C
    length = 2 * math.sqrt(bounding_sphere_r2)
    scale = standard_height / length
    translation = -1 * obj_center
    z_up_translation = internal_rotation @ translation

    T = np.array(
        [[scale, 0, 0, scale * translation[0]],
         [0, scale, 0, scale * translation[1]],
         [0, 0, scale, scale * translation[2]],
         [0, 0, 0, 1]]
    )

    Z_up_T = np.array(
        [[scale, 0, 0, scale * z_up_translation[0]],
         [0, scale, 0, scale * z_up_translation[1]],
         [0, 0, scale, scale * z_up_translation[2]],
         [0, 0, 0, 1]]
    )

    np.savetxt(transform_path, T)
    np.savetxt(z_transform_path, Z_up_T)
    print("Inside transformation calculation; default is y_up")
    print("T is %s" % str(T))
    print("z_up_T is %s" % str(Z_up_T))

    original_mesh.apply_transform(T)

    sdf_results = generate_sdf(original_mesh,
                               space_sample_number=space_sample_number,
                               near_surface_sample_number=near_surface_sample_number,
                               surface_sample_number=surface_sample_number)
    space_data_struct = sdf_results[0]
    near_surface_data_struct = sdf_results[1]
    surface_data_struct = sdf_results[2]

    if shuffle:
        space_data_struct = shuffle_numpy_array(space_sample_number, space_data_struct)
        near_surface_data_struct = shuffle_numpy_array(near_surface_sample_number, near_surface_data_struct)
        surface_data_struct = shuffle_numpy_array(surface_sample_number, surface_data_struct)

    if sample_format == "h5" or sample_format == "h5_chunk":
        h5_path = os.path.join(geometry_sample_folder, "sample.h5")
        with h5py.File(h5_path, "w") as f:
            for data_name in space_data_struct.keys():
                if sample_format == "h5_chunk":
                    print(space_data_struct[data_name].shape)
                    if len(space_data_struct[data_name].shape) <= 1:
                        f.create_dataset(data_name, data=space_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size))
                    else:
                        chunk_size_y = space_data_struct[data_name].shape[1]
                        f.create_dataset(data_name, data=space_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size, chunk_size_y))
                else:
                    f.create_dataset(data_name, data=space_data_struct[data_name], compression="gzip")

            for data_name in near_surface_data_struct.keys():
                if sample_format == "h5_chunk":
                    if len(near_surface_data_struct[data_name].shape) <= 1:
                        f.create_dataset(data_name, data=near_surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size))
                    else:
                        chunk_size_y = near_surface_data_struct[data_name].shape[1]
                        f.create_dataset(data_name, data=near_surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size, chunk_size_y))
                else:
                    f.create_dataset(data_name, data=near_surface_data_struct[data_name], compression="gzip")

            for data_name in surface_data_struct.keys():
                if sample_format == "h5_chunk":
                    if len(surface_data_struct[data_name].shape) <= 1:
                        f.create_dataset(data_name, data=surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size))
                    else:
                        chunk_size_y = surface_data_struct[data_name].shape[1]
                        f.create_dataset(data_name, data=surface_data_struct[data_name], compression="gzip",
                                         chunks=(chunk_size, chunk_size_y))
                else:
                    f.create_dataset(data_name, data=surface_data_struct[data_name], compression="gzip")
    else:
        save_struct_as_numpy(surface_data_struct, output_folder, surface_sample_number)
        save_struct_as_numpy(near_surface_data_struct, output_folder, near_surface_sample_number)
        save_struct_as_numpy(space_data_struct, output_folder, space_sample_number)

    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Curvature based SDF calculation finished. Local time is %s" % (local_time_str))
