
import os,glob,sys
from pathlib import Path
import json

import trimesh
import open3d as o3d
import scipy
import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import spsolve
import scipy.sparse as sparse
from scipy.sparse import vstack

import igl




# ACVD_path = "/apdcephfs/private_rabbityli/workspace/ACVD/bin"
# script_path = "/apdcephfs/private_rabbityli/workspace/clothProxyGeneration/_3_proxy_gen"

def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data

gradation = 0.5


def compute_uniform_laplacian( V, F ):
    L = igl.cotmatrix(V, F)  # laplace-beltrami operator in libigl
    # M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_BARYCENTRIC) # mass matrix in libigl
    # convert cotagent laplacian to uniform laplacian
    L = (np.abs(np.array(L.todense())) > 0) * -1
    D = -1 * L.sum(axis=1) - 1
    np.fill_diagonal(L, D)
    sparse_L = sparse.csc_matrix(L)
    delta = sparse_L.dot(V)
    return sparse_L, delta


def connected_largest_submesh(V, F):

    m = trimesh.Trimesh( V, F)
    sub_graphs = trimesh.graph.connected_components(m.vertex_adjacency_graph.edges)

    #get largest group
    max_g = np.argmax(np.asarray([len(e) for e in sub_graphs]))
    g = sub_graphs[max_g]

    # filter valid triangles
    valid_idx = np.zeros(m.vertices.shape[0])
    valid_idx[g] = 1
    faces = np.array(m.faces)
    faces_mask = valid_idx[faces][..., 0] #reshape(-1)].reshape(-1, 3).sum(-1) == 3
    valid_faces = faces[faces_mask>0]

    # old to new index map
    imap = np.ones( (m.vertices.shape[0])).astype(np.int64)
    imap [g] = np.arange(g.shape[0])
    valid_faces = imap[valid_faces.reshape(-1)].reshape(-1, 3)



    return np.array(m.vertices)[g], valid_faces


def run (proxy_flder, acvd_path):
    manifold_proxy = Path(proxy_flder) / "manifold_single_layer"
    voronoi_proxy = Path(proxy_flder) / "proxy"
    voronoi_proxy.mkdir(exist_ok=True)
    info_json = Path(proxy_flder) / "info.json"
    info = load_json(info_json)
    parts = info["parts"]
    n_verts = info["n_verts"]

    prox_meshes = []


    voronoi_info=[]


    # manifold-1-single_layer.ply
    for idx, nv in enumerate (n_verts):

        m = os.path.join(manifold_proxy, "manifold-" + str(idx) + "-single_layer.ply" )
        proxy_mesh_path = os.path.join( voronoi_proxy, "part-" + str(idx) + ".ply")

        cd_cmd1 = "cd "+ acvd_path + ";"
        cmd = " ".join(
            [
                cd_cmd1,

                "./ACVD",
                m,
                str(nv),
                str(gradation),
                "-of",
                proxy_mesh_path
            ]
        )


        print ("acvd cmd: ", cmd)

        os.system(cmd)



        # load_mesh and filter invalid faces
        proxy_mesh = o3d.io.read_triangle_mesh(proxy_mesh_path)
        proxy_mesh.merge_close_vertices(eps=0.000001)
        num_v_pre = len(proxy_mesh.vertices)
        num_f_pre = len(proxy_mesh.triangles)
        V, F = connected_largest_submesh(V=np.asarray(proxy_mesh.vertices), F=np.asarray(proxy_mesh.triangles))
        proxy_mesh.vertices = o3d.utility.Vector3dVector(V)
        proxy_mesh.triangles = o3d.utility.Vector3iVector(F)
        o3d.io.write_triangle_mesh( proxy_mesh_path, proxy_mesh )
        num_v_pst = len(proxy_mesh.vertices)
        num_f_pst = len(proxy_mesh.triangles)

        print("----------------------------------pre, post----------",num_v_pre,num_f_pre , "/", num_v_pst, num_f_pst,"----------")

        # import pdb; pdb.set_trace()

        # precompute laplacian and save matrix
        proxy_mesh = trimesh.load(proxy_mesh_path)
        L, delta = compute_uniform_laplacian( np.array(proxy_mesh.vertices), np.array(proxy_mesh.faces))
        lfn = os.path.join( voronoi_proxy, "part-" + str(idx) + "-laplacian.npz")
        scipy.sparse.save_npz(lfn, L)

        # prox_meshes
        # proxy_2_visual skinning mapping

        voronoi_info.append( [  num_v_pst, num_f_pst,  len(proxy_mesh.vertices) , len(proxy_mesh.faces)] )


    info[ "voronoi_info_VFVF" ] = voronoi_info


    json_object = json.dumps(info, indent=4)
    with open( info_json, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':


    # fn = "/home/rabbityl/tboard/DR_394_F_A/DR_394_fbx2020.obj"
    #
    # proxy_flder = os.path.join(Path( fn).parent, "proxy_mesh")
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=str, required=True)
    parser.add_argument("--acvd_path", type=str, required=True)
    args = parser.parse_args()

    proxy_flder = args.p
    acvd_path = args.acvd_path
    run(proxy_flder, acvd_path)
