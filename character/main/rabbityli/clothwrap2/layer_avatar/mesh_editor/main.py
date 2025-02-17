import open3d as o3d
import numpy as np
import scipy

from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
from scipy.sparse import vstack

# import igl

def compute_uniform_laplacian( V, F ):
    L = igl.cotmatrix(V, F)  # laplace-beltrami operator in libigl
    # M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC) # mass matrix in libigl
    # convert cotagent laplacian to uniform laplacian
    L = (np.abs(np.array(L.todense())) > 0) * -1
    D = -1 * L.sum(axis=1) - 1
    np.fill_diagonal(L, D)
    L = sparse.csc_matrix(L)
    delta = L.dot(V)
    return L, delta


from pene_solve import solve_pene

if __name__ == '__main__':

    proxy_mesh = "/home/rabbityl/workspace/clothPenSolve/proxies/bottom.obj"
    proxy_mesh = o3d.io.read_triangle_mesh(proxy_mesh)
    proxy_mesh.merge_close_vertices(eps=0.000001)
    #
    # proxy_mesh2 = "./proxies/top.obj"
    # proxy_mesh2 = o3d.io.read_triangle_mesh(proxy_mesh2)
    # proxy_mesh2.merge_close_vertices(eps=0.000001)

    smplx_mesh = "/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/quest_male_slim/warpped_smpl.obj"
    smplx_mesh = o3d.io.read_triangle_mesh(smplx_mesh)
    R = smplx_mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    smplx_mesh.rotate(R, center=(0, 0, 0))

    V = np.asarray(proxy_mesh.vertices)
    F = np.asarray(proxy_mesh.triangles)
    L, delta = compute_uniform_laplacian(V, F)

    res = solve_pene(smplx_mesh, proxy_mesh, L, delta, PEN_THRESH=0.0005, PUSH_DIST=0.015, PUSH_WEIGHT=100, HOLD_WEIGHT=1)

    if res is not None:
        proxy_mesh.vertices = o3d.utility.Vector3dVector(res)



    o3d.visualization.draw( [ proxy_mesh, smplx_mesh])
    # proxy_mesh = solver.solve(proxy_mesh, smplx_mesh,n_iter=1 , vis=True)





    # o3d.visualization.draw([proxy_mesh,proxy_mesh2, smplx_mesh])

    # o3d.visualization.draw([proxy_mesh, smplx_mesh])
