import copy

import open3d as o3d
import numpy as np
import scipy

from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
from scipy.sparse import vstack

# import igl



PENETRATION_THRESHOLD = 0.0005
PUSH_DISTANCE = 0.015



def solve( A, b):
    '''
    :param A: sparse CSC matrix [m,n]
    :param b: array [m,3]
    :return: x: array [n,3]
    '''
    x = []
    for i in range(3):
        x.append(lsqr(A, b[:, i])[0])
    x = np.stack(x, axis=-1).astype(np.float32)
    return x


def construct_smooth_system(proxy_verts, vert_mask, L, delta, CTR_WEIGHT = 1.0):

    pts_index = np.arange(len(proxy_verts))

    if vert_mask:
        ctr_index = pts_index[vert_mask]
        ctrl_pts = proxy_verts[vert_mask]
    else:
        ctr_index = pts_index
        ctrl_pts = proxy_verts

    row = np.arange(len(ctrl_pts))
    col = ctr_index
    val = np.array([CTR_WEIGHT] * len(ctrl_pts))

    control_block = sparse.csc_matrix((val, (row, col)), shape=(len(ctrl_pts), len(proxy_verts)))
    b = ctrl_pts * CTR_WEIGHT

    A = vstack([L, control_block])
    b = np.concatenate([delta, b], axis=0)

    return A, b


def construct_penetration_resolve_system(proxy_verts, vert_mask, L, delta, CTR_WEIGHT = 1.0):

    pts_index = np.arange(len(proxy_verts))

    if vert_mask:
        ctr_index = pts_index[vert_mask]
        ctrl_pts = proxy_verts[vert_mask]
    else:
        ctr_index = pts_index
        ctrl_pts = proxy_verts

    row = np.arange(len(ctrl_pts))
    col = ctr_index
    val = np.array([CTR_WEIGHT] * len(ctrl_pts))

    control_block = sparse.csc_matrix((val, (row, col)), shape=(len(ctrl_pts), len(proxy_verts)))
    b = ctrl_pts * CTR_WEIGHT

    A = vstack([L, control_block])
    b = np.concatenate([delta, b], axis=0)

    return A, b








def concatenent_meshes(V_in, F_in):

    V_lst = copy.deepcopy(V_in)
    F_lst = copy.deepcopy(F_in)

    assert len(V_lst) == len(F_lst)
    assert len(V_lst) >=1

    if len(V_lst) > 1 :
        size = [len(e) for e in V_lst[:-1]]
        id_offset = [0] + list(np.cumsum(size))
        for i in range(len(F_lst)):
            F_lst[i] = F_lst[i] + id_offset[i]
        V = np.concatenate(V_lst, axis=0)
        F = np.concatenate(F_lst, axis=0)
        return V, F
    else:
        return V_lst[0], F_lst[0]





class PenetrationSolver:

    # def __init__(self ): pass

    HOLD_WEIGHT = 1
    PUSHOUT_WEIGHT = 100
    body_scene = None

    def solve(self, proxy_mesh, body_mesh, n_iter=1, vis =False ):

        # Create a scene and add the triangle mesh
        # mesh = o3d.t.geometry.TriangleMesh.from_legacy(smplx_mesh)
        # scene = o3d.t.geometry.RaycastingScene()
        # _ = scene.add_triangles(mesh)

        if self.body_scene is None:
            self.register_body(body_mesh)


        V = np.asarray(proxy_mesh.vertices)
        F = np.asarray(proxy_mesh.triangles)
        L, delta = compute_uniform_laplacian(V, F)


        for i_ter in range(n_iter):

            print( "iter:", i_ter)

            # compute sdf
            sdf, _, face_ids = compute_signed_distance_and_closest_goemetry(
                np.asarray(proxy_mesh.vertices).astype(np.float32), self.body_scene)
            is_inside = sdf < PENETRATION_THRESHOLD

            proxy_verts = np.array(proxy_mesh.vertices)


            if vis: # show collision
                color = np.ones_like(proxy_verts) * 0.5
                color[is_inside] = np.asarray([0.9, 0.1, 0.1])
                proxy_mesh.vertex_colors = o3d.utility.Vector3dVector(color)
                proxy_mesh.compute_vertex_normals()
                # o3d.visualization.draw([proxy_mesh, smplx_mesh])


            # compute offset
            face_normals = self.body_normals[face_ids][is_inside]
            distance = sdf[is_inside] + PENETRATION_THRESHOLD + PUSH_DISTANCE
            offset = distance[..., None] * face_normals
            proxy_verts[is_inside] = proxy_verts[is_inside] + offset

            num_inside = is_inside.sum()

            #construct A@x=b
            pts_index = np.arange(len(proxy_verts))

            if  num_inside >0:
                ctr_index = pts_index [is_inside]
                ctrl_pts = proxy_verts [is_inside]
                row = np.arange(len(ctrl_pts))
                col = ctr_index
                val = np.array([self.PUSHOUT_WEIGHT] * len(ctrl_pts))
                control_block = sparse.csc_matrix((val, (row, col)), shape=(len(ctrl_pts), len(V)))
                b = ctrl_pts * self.PUSHOUT_WEIGHT

                if len(proxy_verts) - num_inside > 0:
                    hold_index = pts_index[~is_inside]
                    hold_pts = proxy_verts[~is_inside]

                    row = np.arange(len(hold_pts))
                    col = hold_index
                    val = np.array([self.HOLD_WEIGHT] * len(hold_pts))
                    hold_block = sparse.csc_matrix((val, (row, col)), shape=(len(hold_pts), len(V)))
                    b2 = hold_pts * self.HOLD_WEIGHT

                    A = vstack([L, control_block, hold_block])
                    b = np.concatenate([delta, b, b2], axis=0)

                else:


                    A = vstack([L, control_block])
                    b = np.concatenate([delta, b], axis=0)



                x = solve(A, b)
                #
                # x = []
                # for i in range(3):
                #     x.append(lsqr(A, b[:, i])[0])
                # x = np.stack(x, axis=-1).astype(np.float32)


        if vis:
            proxy_mesh.vertices = o3d.utility.Vector3dVector(x)
            o3d.visualization.draw([proxy_mesh, smplx_mesh])



        # return  proxy_mesh







def pensolve():

    proxy_mesh = "./proxies/bottom.obj"
    proxy_mesh = o3d.io.read_triangle_mesh(proxy_mesh)
    proxy_mesh.merge_close_vertices(eps=0.000001)

    # proxy_mesh2 = "./proxies/top.obj"
    # proxy_mesh2 = o3d.io.read_triangle_mesh(proxy_mesh2)
    # proxy_mesh2.merge_close_vertices(eps=0.000001)

    smplx_mesh = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/quest_male_slim/warpped_smpl.obj"
    smplx_mesh = o3d.io.read_triangle_mesh(smplx_mesh)
    R = smplx_mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    smplx_mesh.rotate(R, center=(0, 0, 0))



    solver =  PenetrationSolver()

    proxy_mesh = solver.solve(proxy_mesh, smplx_mesh,n_iter=1 , vis=True)

    # o3d.visualization.draw([proxy_mesh, proxy_mesh2, smplx_mesh])
    # o3d.visualization.draw([proxy_mesh, smplx_mesh])


def main():

    proxy_mesh = "/home/rabbityl/tboard/DR_394_F_A/proxy_mesh/voronoi_proxy/part-1.ply"
    proxy_mesh = o3d.io.read_triangle_mesh(proxy_mesh)
    proxy_mesh.merge_close_vertices(eps=0.000001)
    proxy_mesh.paint_uniform_color( [1, 0.1, 0])


    V = np.asarray(proxy_mesh.vertices)
    F = np.asarray(proxy_mesh.triangles)
    L, delta = compute_uniform_laplacian(V, F)

    # np.random
    random_offseted_V = V + (np.random.rand( V.shape[0],V.shape[1])   ) * 0.01 + 0.02
    proxy_mesh3 = copy.deepcopy(proxy_mesh)
    proxy_mesh3.vertices = o3d.utility.Vector3dVector ( random_offseted_V )
    proxy_mesh3.paint_uniform_color( [0.1, 0, 1])



    A,b = construct_smooth_system( random_offseted_V, None, L, delta, 100)

    x = solve(A,b)

    proxy_mesh2 = copy.deepcopy(proxy_mesh)
    proxy_mesh2.vertices = o3d.utility.Vector3dVector ( x )
    proxy_mesh2.paint_uniform_color( [0.1, 1, 0])

    proxy_mesh.compute_vertex_normals()
    proxy_mesh3.compute_vertex_normals()
    proxy_mesh2.compute_vertex_normals()

    o3d.visualization.draw([proxy_mesh, proxy_mesh3, proxy_mesh2])