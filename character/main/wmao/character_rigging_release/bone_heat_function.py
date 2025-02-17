import numpy as np
from scipy.linalg import cholesky, solve
import trimesh 
from ipdb import set_trace as st
from pysdf import SDF
import open3d as o3d
from matplotlib import cm
import time
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv as spinv

from threading import Thread, current_thread
import multiprocessing
from multiprocessing import Process, current_process
from concurrent.futures import ProcessPoolExecutor
import os
import copy

def distsq_to_seg_batch(v, p1, p2):
    """
    Compute the squared distance from point v to the line segment defined by p1 and p2.

    Args:
        v (np.ndarray)(nv, 3): A vector representing the point in space.
        p1 (np.ndarray)(3): The first endpoint of the line segment.
        p2 (np.ndarray)(3): The second endpoint of the line segment.

    Returns:
        float: The squared distance from the point v to the closest point on the segment [p1, p2].
    """


    vp = np.zeros_like(v)
    dist = np.zeros(v.shape[0]) - 1

    dir = p2[None] - p1[None]  # Direction vector of the line segment
    difp2 = p2[None] - v  # Vector from point v to the second endpoint p2
    idx = np.sum(difp2*dir,axis=1) <= 0
    vp[idx] = p2
    dist[idx] = np.sum(difp2[idx]*difp2[idx],axis=1)

    difp1 = v - p1[None]  # Vector from point v to the first endpoint p1
    dot = np.sum(difp1*dir,axis=1)
    vp[dot<=0] = p1
    dist[dot<=0] = np.sum(difp1[dot<=0]*difp1[dot<=0],axis=1) # Squared distance to p1 if v is before p1
    idx_remain = np.logical_not(np.logical_or(idx,dot<=0))
    
    # Compute the squared distance to the line segment
    vp[idx_remain] = p1[None] + dir * (dot[idx_remain,None] / (np.sum(dir*dir) + 1e-10))
    dist[idx_remain] = np.sum(difp1[idx_remain]*difp1[idx_remain], axis=1) - (dot ** 2)[idx_remain] / (np.sum(dir*dir) + 1e-10)
    return vp, dist

def compute_bone_dist(inps):
    stime = time.time()
    pid = os.getpid()
    threadName = current_thread().name
    processName = current_process().name
    print(f"{pid} * {processName} * {threadName} \
        ---> Start counting...")
    v1, v2, verts, verts_smp, faces_smp, npts = inps
    # v1, v2 = joints[parents[j]], joints[j]
    f = SDF(verts_smp, faces_smp)
    vp, sqd = distsq_to_seg_batch(verts, v1, v2)
    bone_dists = np.sqrt(sqd)
    direction = verts - vp
    sampled_points = np.linspace(0, 0.9999, npts)[:, None, None] * direction[None] + vp[None]
    is_start_inside = f.contains(sampled_points.reshape([-1,3])).reshape([npts, -1])
    bone_vis = np.logical_and(is_start_inside.sum(axis=0)>=npts, bone_dists < 0.5)

    print(f"{pid} * {processName} * {threadName} used {time.time()-stime:.3f} \
        ---> Finished counting...")
    return bone_vis, bone_dists

def convert2float32(inps):
    out = []
    for inp in inps:
        if inp.dtype == np.float64:
            out.append(inp.astype('float32'))
        else:
            out.append(inp)
    return out

def sparse_linear_solver(inps):
    Asp, rhs = inps
    solution = spsolve(Asp, rhs)
    return np.clip(solution, a_min=1e-10, a_max=1.0)

def bone_heat_algorithm(verts, faces, joints, parents, verts_smp, faces_smp, initial_heat_weight=1):
    """
    compute the blending weights using bone heat algorithm. 
    verts: nv x 3
    faces: nf x 3
    joints: nj x 3
    parents: nj
    initial_heat_weight: 1
    """
    verts,joints,verts_smp = convert2float32([verts,joints,verts_smp])
    nv = verts.shape[0]
    nj = joints.shape[0]
    f = SDF(verts_smp, faces_smp)
    
    # define sparse version
    stime = time.time()
    rows = np.concatenate([faces[:,0],faces[:,1],faces[:,0],faces[:,2],faces[:,1],faces[:,2]])
    cols = np.concatenate([faces[:,1],faces[:,0],faces[:,2],faces[:,0],faces[:,2],faces[:,1]])
    values = np.ones(cols.shape[0], dtype=np.float32)
    # Create a COO sparse matrix
    adj = coo_matrix((values, (rows, cols)), shape=(nv, nv)).tocsr()
    adjdegree = np.array(adj.sum(axis=1))[:,0]
    values = values / adjdegree[rows]
    adj = coo_matrix((values, (rows, cols)), shape=(nv, nv)).tocsr()
    print(f'define adj used {time.time()-stime:.3f}')
    
    # Calculate bone distances and visibility
    stime = time.time()
    bone_dists = np.zeros([nv, nj-1],dtype=np.float32) - 1
    bone_vis = np.zeros([nv, nj-1],dtype=np.float32) > 0 
    npts = 50
    for j in range(1, nj):
        v1, v2 = joints[parents[j]], joints[j]
        vp, sqd = distsq_to_seg_batch(verts, v1, v2)
        bone_dists[:, j-1] = np.sqrt(sqd)
        direction = verts - vp
        sampled_points = np.linspace(0, 0.9999, npts,dtype=np.float32)[:, None, None] * direction[None] + vp[None]
        is_start_inside = f.contains(sampled_points.reshape([-1,3])).reshape([npts, -1]) # this is not very stable
        bone_vis[:, j-1] = np.logical_and(is_start_inside.sum(axis=0)>=npts, bone_dists[:, j-1] < 0.5)
    print(f'bone dist used {time.time() - stime:.3f}')

    # stime = time.time()
    # bone_dists1 = np.zeros([nv, nj-1],dtype=np.float32) - 1
    # bone_vis1 = np.zeros([nv, nj-1],dtype=np.float32) > 0 
    # npts = 50
    # inputs = [(joints[parents[j]], joints[j], verts, verts_smp, faces_smp, npts) for j in range(1, nj)]
    # with multiprocessing.Pool(processes=min(nj-1, multiprocessing.cpu_count())) as pool:
    #     results = pool.map(compute_bone_dist, inputs)
    # # for j in range(nj-1):
    # #     bone_vis1[:,j] = results[j][0]
    # #     bone_dists1[:,j] = results[j][1]
    # print(f'bone dist multiprocessing used {time.time() - stime:.3f}')
    
    # handle vertices that is not visible to any bone
    stime = time.time()
    while True:
        vertex_invis = np.where(np.logical_not(np.any(bone_vis,axis=1)))[0]
        if len(vertex_invis) == 0:
            break
        tmp = bone_vis * 1.0
        tmp = adj @ tmp
        tmp = tmp > 0
        bone_vis[vertex_invis] = tmp[vertex_invis]
    # smooth the visibility
    tmp = bone_vis * 1.0
    tmp = adj @ tmp + tmp
    tmp = tmp > 0.49
    bone_vis = tmp
    bone_vis = np.array(bone_vis)
    print(f'bone dist refine used {time.time() - stime:.3f}')
    
    # We have -Lw+Hw=HI, same as (H-L)w=HI
    # H is the minimal distance from a vertex to the bone
    # L is the mesh laplacian matrix
    # I is bone specific is whether a bone is visible to the vertex
    stime = time.time()
    min_dist = np.min(bone_dists + (1-bone_vis) * 10000, axis=1)
    closest = np.argmin(bone_dists + (1-bone_vis) * 10000, axis=1)
    H = initial_heat_weight / (1e-6 + min_dist**2)
    
    # compute face areas
    v1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    v2 = verts[faces[:, 1]] - verts[faces[:, 0]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1, v2),axis=1)
    rows = faces.reshape(-1)
    cols = np.zeros_like(rows)
    values = face_areas[np.arange(rows.shape[0])//3]
    vert_areas = coo_matrix((values,(rows,cols)),shape=(nv,1)).toarray()[:,0]
    print(f'compute vert area used {time.time() - stime:.3f}')

    # vert_areas = np.zeros(nv)
    # for vi in range(nv):
    #     vert_areas[vi] = np.sum(face_areas[np.any(faces==vi, axis=1)])

    stime = time.time()
    # compute per edge cot
    edge_cot = np.zeros([faces.shape[0], 3],dtype=np.float32)
    # L = np.zeros([nv, nv])
    # make sure that scipy will add the values when an element is assigned multiple times
    rows = []
    cols = []
    values = []
    for i in range(3):
        v1 = verts[faces[:,(i+1)%3]] - verts[faces[:, i]]
        v2 = verts[faces[:,(i+2)%3]] - verts[faces[:, i]]
        edge_cot[:,i] = np.sum(v1*v2, axis=1) / (1e-6 + np.linalg.norm(np.cross(v1, v2), axis=1))
        # L[faces[:,(i+1)%3], faces[:,(i+2)%3]] += 0.5 * edge_cot[:, i]
        # L[faces[:,(i+2)%3], faces[:,(i+1)%3]] += 0.5 * edge_cot[:, i]
        rows.append(faces[:,(i+1)%3])
        cols.append(faces[:,(i+2)%3])
        values.append(0.5 * edge_cot[:, i])
        rows.append(faces[:,(i+2)%3])
        cols.append(faces[:,(i+1)%3])
        values.append(0.5 * edge_cot[:, i])
    # L = L - np.diag(L.sum(axis=1))
    # L = 3 * L / (vert_areas[:,None] + 1e-6)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    values = np.concatenate(values)
    L = coo_matrix((values, (rows, cols)), shape=(nv, nv)).tocsr()
    rows2, cols2 = np.arange(nv), np.arange(nv)
    values2 = np.array(L.sum(axis=1))[:,0]
    # L_diag = coo_matrix((values, (rows2, cols2)), shape=(nv, nv)).tocsr()
    rows = np.concatenate([rows, rows2])
    cols = np.concatenate([cols, cols2])
    values = np.concatenate([values, -1 * values2])
    values = 3 * values / (vert_areas[rows] + 1e-6)
    L = coo_matrix((values, (rows, cols)), shape=(nv, nv)).tocsr()
    # L = L - L_diag
    # L = 3 * L / (vert_areas[:,None] + 1e-6)
    # L = csr_matrix(L)
    print(f'compute laplacian matrix used {time.time() - stime:.3f}')

    stime = time.time()
    # Solve linear system using factorization
    rows, cols = np.arange(nv), np.arange(nv)
    H_sp = coo_matrix((H, (rows, cols)), shape=(nv, nv)).tocsr()
    Asp = H_sp - L
    nz_weights = np.zeros([nv, nj-1], dtype=np.float32)
    for j in range(nj-1):
        # rhs = H * (closest == j)
        rows = np.where(closest == j)[0]
        cols = np.zeros_like(rows)
        values = H[rows]
        rhs = coo_matrix((values, (rows, cols)), shape=(nv, 1)).tocsr()
        # Solve the system Ax = b
        solution = spsolve(Asp, rhs)
        # solution = np.dot(Ainv, rhs)
        # nz_weights[:,j] = solution 
        nz_weights[:,j] = np.clip(solution, a_min=1e-10, a_max=1.0)
    weights = nz_weights / nz_weights.sum(axis=1)[:,None]
    print(f'compute weights used {time.time() - stime:.3f}')

    # stime = time.time()
    # inputs = [(Asp, H * (closest == j)) for j in range(nj-1)]
    # with multiprocessing.Pool(processes=min(nj-1, multiprocessing.cpu_count())) as pool:
    #     results = pool.map(sparse_linear_solver, inputs)
    # for j in range(nj-1):
    #     nz_weights[:,j] = results
    # weights1 = nz_weights / nz_weights.sum(axis=1)[:,None]
    # print(f'compute weights multiprocessing used {time.time() - stime:.3f}')

    return weights, H, vert_areas, L, closest


def attachment_private(mesh_dir, skeleton_npz, initial_heat_weight=1, out_dir='', is_debug=True):
    mesh = trimesh.load(mesh_dir)
    nv = mesh.vertices.shape[0]
    verts = mesh.vertices
    vmax = verts.max(axis=0)
    vmin = verts.min(axis=0)
    # (vmax - vmin)
    # scale = 2 / np.linalg.norm(vmax-vmin)
    # verts = scale * verts
    faces = mesh.faces


    mesh_smp = trimesh.load('/aigc_cfs_gdp/weimao/character_rigging/test_data/mesh_rot_manifold_simp.obj')
    verts_smp = mesh_smp.vertices
    faces_smp = mesh_smp.faces
    
    parents = [-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 0, 17, 18]
    nb = len(parents)
    skeleton = np.load(skeleton_npz)['joints']

    weights, H, vert_areas, L, closest = bone_heat_algorithm(verts, faces, skeleton, parents, verts_smp, faces_smp)
    if is_debug:
        cmap = cm.get_cmap('jet')
        # save results
        for j in range(nb-1):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            col = cmap(weights[:,j])[:,:3]
            mesh.vertex_colors = o3d.utility.Vector3dVector(col)
            o3d.io.write_triangle_mesh(f'{out_dir}/{j:03d}.ply', mesh)

        cmap = cm.get_cmap('tab20')

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        col = cmap(closest)[:,:3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(col)
        o3d.io.write_triangle_mesh(f'{out_dir}/parts.ply', mesh)
    np.savez_compressed(f'{out_dir}/weights.npz',weights=weights, verts=verts, faces=faces, joints=skeleton, parents=parents)
    return weights


if __name__=="__main__":
    stime = time.time()
    weights = attachment_private("/aigc_cfs_gdp/weimao/character_rigging/test_data/mesh_rot_manifold.obj", 
                                "/aigc_cfs_gdp/weimao/character_rigging/test_data/joint3d_v2.npz", 
                                out_dir='/aigc_cfs_gdp/weimao/character_rigging/output/',
                                is_debug=True)
    print(f'time used {time.time() - stime:.3f}')