import copy
import os.path
import trimesh
import open3d as o3d
import trimesh
import torch
import numpy as np
import argparse
import json
import pathlib






def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data


def connected_largest_submesh(V, F):

    m = trimesh.Trimesh( V, F)
    sub_graphs = trimesh.graph.connected_componentsbody_path(m.vertex_adjacency_graph.edges)

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

    valid_verts = V
    # valid_verts = V[g]

    return np.array(m.vertices)[g], valid_faces


def process(visual_path, proxy_flder, vert_limit, body_path=None):



        



    #load visual mesh
    visual = o3d.io.read_triangle_mesh(visual_path)
    visual.paint_uniform_color( [0, 1, 0])

    # sample densely on visual mesh
    visual_points = visual.sample_points_uniformly(number_of_points = 500000 )
    visual_points.paint_uniform_color([0, 0, 1])

    # load manifold meshes
    blender_info = load_json( os.path.join(proxy_flder, "info.json") )
    thinkness = blender_info["thinkness"]
    parts = blender_info["parts"]
    parts = [os.path.join(proxy_flder, "manifold", "manifold-" + str(i) + ".obj") for i in range (parts)]
    manifold_meshes = []
    n_faces = []
    total_face_num = 0.0
    for manifold_path in parts:
        #load manifold mesh
        manifd = o3d.io.read_triangle_mesh( manifold_path )
        manifd.paint_uniform_color( [1,0,0])
        manifd.merge_close_vertices(eps=0.000001)
        total_face_num += len(manifd.triangles)
        n_faces.append( len(manifd.triangles) )
        manifold_meshes.append( manifd)



    # import pdb; pdb.set_trace()

    # assign faces
    n_verts = [int(vert_limit * e / total_face_num) for e in n_faces]

    manifold_proxy_flder = pathlib.Path(proxy_flder) / "manifold_single_layer"
    manifold_proxy_flder.mkdir(exist_ok=True)

    for idx, manifd in enumerate ( manifold_meshes):

        manifold_subset = near_visual_submanifold( manifd, visual_points , thinkness)

        part_path = os.path.join(manifold_proxy_flder, "manifold-"+str(idx)+"-single_layer.ply")

        o3d.io.write_triangle_mesh( part_path, manifold_subset)


    blender_info["n_verts"] = n_verts


    blender_info_path = os.path.join(proxy_flder, "info.json")
    json_object = json.dumps(blender_info, indent=4)
    with open( blender_info_path, "w") as outfile:
        outfile.write(json_object)


def near_visual_submanifold( manifd, visual_points , thinkness):

    #compute face center of manifold mesh
    manifd_verts = np.asarray(manifd.vertices).astype(np.float32)
    manifd_faces = np.asarray(manifd.triangles)
    face_center = np.sum(manifd_verts[manifd_faces], axis=1).squeeze() / 3
    pc_manifd = o3d.geometry.PointCloud()
    pc_manifd.points = o3d.utility.Vector3dVector(face_center)
    pc_manifd.paint_uniform_color([0.8, 0.1, 0.1])

    #compute distance manifold to visual
    pcd_tree = o3d.geometry.KDTreeFlann(visual_points)
    distances = []
    for i, point in enumerate(pc_manifd.points):
        [count, vec1, vec2] = pcd_tree.search_knn_vector_3d(point, 1)
        distances.append(np.sqrt(vec2[0]))
    valid_face_mask = np.asarray(distances) < thinkness * 0.9

    #Remove invalid faces
    manifd_subset = copy.deepcopy(manifd)
    manifd_subset.remove_triangles_by_mask(~valid_face_mask)

    manifd_subset.merge_close_vertices(eps=0.000001)

    V, F = connected_largest_submesh( V=np.asarray(manifd_subset.vertices), F=np.asarray(manifd_subset.triangles))
    manifd_subset.vertices = o3d.utility.Vector3dVector(V)
    manifd_subset.triangles = o3d.utility.Vector3iVector(F)
    return manifd_subset


def obtain_body_mesh():
    Asset_key = "BTM_382"
    body_key = ["female", "mcwy2"]
    body_path = "_2_single_layer_manifold/MCWY2_F_T/smplx_and_offset_smplified.npz"

    if body_path is not None:
        param_data = torch.load(body_path, map_location='cuda:0')
        smpl_faces = param_data['faces'].detach().cpu().numpy()
        smpl_verts = param_data['posed_verts'].detach().cpu().numpy()


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=str, required=True)
    parser.add_argument("--p", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--body_path", type=str, default=None)
    args = parser.parse_args()

    visual_path = args.v
    proxy_flder = args.p
    vert_limit = args.n
    body_path = args.body_path






    process( visual_path=visual_path, proxy_flder=proxy_flder, vert_limit=vert_limit, body_path=body_path)

    # process( visual_path="/home/rabbityl/tboard/DR_394_F_A/DR_394_fbx2020.obj" ,vert_limit=6000)


