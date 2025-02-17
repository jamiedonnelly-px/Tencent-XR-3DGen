import argparse

import os
import json
import numpy as np
import  open3d as o3d
import matplotlib
from pathlib import Path
cmap = matplotlib.cm.get_cmap('viridis')

mirror_keys = [
    "arm1_ring",
    "arm2_ring",
    "arm3_ring",
    "leg1_ring",
    "leg2_ring",
    "leg3_ring",
    "foot_face",
    "foot_bottom_end",
    "foot_bottom_front"
]

def curve_len( edges, verts ):
    l = 0
    for e in edges:
        l += np.linalg.norm( verts[e[0]] - verts[e[1]] )
    return l

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def sample_points_on_polyline(edges, verts,  num_points):

    global cmap
    edges = np.array(edges)
    # Calculate the total length of the polyline
    total_length = 0
    edge_lengths = []
    for e in edges:
        e_len = np.linalg.norm(verts[e[0]] - verts[e[1]])
        total_length += e_len
        edge_lengths.append( e_len )
    edge_lengths = np.array(edge_lengths)
    edge_lengths[-1] = edge_lengths[-1] + 0.0001 # prevent float number leak
    cum_len = np.cumsum(edge_lengths)

    ratio_curve = np.arange(num_points) / (num_points-1)
    distance = ratio_curve * total_length
    edge_ids = np.argmax ((cum_len[None]-distance[...,None]) >= 0, axis=1)
    ratio_edge = (cum_len [ edge_ids ] - distance) / edge_lengths [ edge_ids ]
    sample_edge = edges[edge_ids]
    sample_points = verts[sample_edge[:,0]] * ratio_edge[...,None] + verts[sample_edge[:,1]] * (1 - ratio_edge[...,None])
    pts_color = cmap( ratio_curve )[...,:3]

    return sample_points, pts_color, ratio_edge, sample_edge

def edge_to_triangle(  edge_vids, edge_baryc ):
    # append dummy 3rd vert
    dummy_vert = np.zeros( [edge_vids.shape[0] ], dtype=int)  # Assume vertex 0 is not in any curve
    dummy_weight = np.zeros( [edge_vids.shape[0] ])
    triangle_vids = np.concatenate( [edge_vids, dummy_vert[..., None]], axis=1 )
    tirangle_baryc = np.stack( [edge_baryc, 1 - edge_baryc, dummy_weight], axis=-1 )
    return triangle_vids, tirangle_baryc


def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_mesh_npy", type=str, required=True)
    parser.add_argument("--src_curves_json", type=str, required=True)
    parser.add_argument("--tgt_mesh_npy", type=str, required=True)
    parser.add_argument("--tgt_curves_json", type=str, required=True)
    args = parser.parse_args()




    with open(args.src_mesh_npy, 'rb') as f:
        smpl_vert_data = np.load(f)
        smpl_face_data = np.load(f)
        smplmesh = o3d.geometry.TriangleMesh()
        smplmesh.vertices = o3d.utility.Vector3dVector(smpl_vert_data)
        smplmesh.triangles = o3d.utility.Vector3iVector(smpl_face_data)
        smplmesh.compute_vertex_normals()
    smpl_curve = load_json( args.src_curves_json)


    with open(args.tgt_mesh_npy, 'rb') as f:
        ym_vert_data = np.load(f)
        ym_face_data = np.load(f)
        yuanmeng_mesh = o3d.geometry.TriangleMesh()
        yuanmeng_mesh.vertices = o3d.utility.Vector3dVector(ym_vert_data)
        yuanmeng_mesh.triangles = o3d.utility.Vector3iVector(ym_face_data)
        yuanmeng_mesh.compute_vertex_normals()
    yuanmeng_curve = load_json( args.tgt_curves_json )



    # compute the length of each curve
    curve_match = { }
    for k in yuanmeng_curve.keys() :
        if k in smpl_curve:
            e_ym = yuanmeng_curve [k] [ "edge"]
            e_sp = smpl_curve [k] [ "edge"]
            if len( e_ym ) > 0 and len( e_sp ) > 0 :
                curve_match[k] = {
                    "curve_len": 0,
                }
                print("---edge valid---", k)
                curve_match[k]["curve_len"] = curve_len(e_ym, ym_vert_data)
                # if k in mirror_keys: # double the weights
                #     curve_match[k]["curve_len"] = curve_match[k]["curve_len"] * 2
            else:
                print("---edge no valid---", k)


    # LOAD smpl mirror map
    # with open ("../point_match/smpl_left_right_symmetric_map.npy", "rb") as f :
    #     mirror_smpl_vert_map = np.load(f)


    #assign point on each curve

    jdata = {
        "character_pts": [],
        "smpl_baryc_coords": [],
        "smpl_verts_id": []
    }
    smpl_clr = []
    smpl_pts = []
    ym_clr = []

    n_points = 10000
    total_len = 0
    for k in curve_match:
        total_len += curve_match[ k ]["curve_len"]
    for k in curve_match:
        curve_match[k]["num_pts"] = int( n_points * curve_match[k]["curve_len"] / total_len)
        curve_match[k]["mathces"] = {
            "character_pts": [],
            "smpl_baryc_coords": [],
            "smpl_verts_id": []
        }

        ym_pts, ym_pts_color, _,_ = sample_points_on_polyline( yuanmeng_curve[k]["edge"],  ym_vert_data, curve_match[k]["num_pts"])
        sp_pts, sp_pts_color, edge_baryc, edge_vids = sample_points_on_polyline( smpl_curve[k]["edge"],  smpl_vert_data, curve_match[k]["num_pts"])
        sp_vids, sp_baryc  = edge_to_triangle(  edge_vids, edge_baryc )



        # if k in mirror_keys: # flip the curve
        #     mirror_sp_pts = sp_pts.copy()
        #     mirror_sp_pts[:, 0] = mirror_sp_pts[:, 0] * -1
        #     mirror_sp_vids = mirror_smpl_vert_map[1][sp_vids.reshape(-1)].reshape(-1, 3)
        #     mirror_sp_baryc = sp_baryc.copy()
        #
        #     sp_pts = np.concatenate([sp_pts, mirror_sp_pts], axis=0)
        #     sp_pts_color = np.concatenate([sp_pts_color, sp_pts_color], axis=0)
        #     sp_baryc = np.concatenate([sp_baryc, mirror_sp_baryc], axis=0)
        #     sp_vids = np.concatenate([sp_vids, mirror_sp_vids], axis=0)
        #
        #
        #     mirror_ym_pts = ym_pts.copy()
        #     mirror_ym_pts[:, 0] = mirror_ym_pts[:, 0] * -1
        #     ym_pts = np.concatenate([ym_pts, mirror_ym_pts], axis=0)
        #     ym_pts_color = np.concatenate([ym_pts_color, ym_pts_color], axis=0)
        #
        #
        #     # verts_id = np.concatenate([verts_id, mirror_verts_id], axis=0)
        #     # baryc_coords = np.concatenate([baryc_coords, mirror_baryc_coords], axis=0)
        #     # src_pts = np.concatenate([src_pts, mirror_src_pts], axis=0)
        #
        #     viz = False
        #     if viz :
        #         smpl_pc = o3d.geometry.PointCloud()
        #         smpl_pc.points = o3d.utility.Vector3dVector( sp_pts )
        #         smpl_pc.colors = o3d.utility.Vector3dVector( sp_pts_color )
        #
        #         ym_pc = o3d.geometry.PointCloud()
        #         ym_pc.points = o3d.utility.Vector3dVector( ym_pts )
        #         ym_pc.colors = o3d.utility.Vector3dVector( ym_pts_color )
        #
        #         # smpl_pc.paint_uniform_color([1, 0, 0])
        #         # o3d.visualization.draw([yuanmeng_mesh, ym_pc, smplmesh, smpl_pc])
        #
        #         o3d.visualization.draw([smplmesh, smpl_pc])
        #         # print("key")


        jdata["character_pts"].append(ym_pts)
        jdata["smpl_baryc_coords"].append(sp_baryc)
        jdata["smpl_verts_id"].append(sp_vids)
        smpl_pts.append( sp_pts )


        smpl_clr.append(sp_pts_color)
        ym_clr.append(ym_pts_color)


    jdata["character_pts"] = np.concatenate( jdata["character_pts"] , axis=0 )
    jdata["smpl_baryc_coords"] = np.concatenate( jdata["smpl_baryc_coords"], axis=0 )
    jdata["smpl_verts_id"] = np.concatenate( jdata["smpl_verts_id"], axis=0 )
    smpl_pts = np.concatenate( smpl_pts, axis= 0 )

    smpl_clr = np.concatenate( smpl_clr, axis= 0 )
    ym_clr = np.concatenate( ym_clr, axis= 0 )


    viz = True
    if viz:
        smpl_pc = o3d.geometry.PointCloud()
        smpl_pc.points = o3d.utility.Vector3dVector(smpl_pts)
        smpl_pc.colors = o3d.utility.Vector3dVector(smpl_clr)

        ym_pc = o3d.geometry.PointCloud()
        ym_pc.points = o3d.utility.Vector3dVector(jdata["character_pts"])
        ym_pc.colors = o3d.utility.Vector3dVector(ym_clr)

        # smpl_pc.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw([yuanmeng_mesh, ym_pc, smplmesh, smpl_pc])

        # o3d.visualization.draw([smplmesh, smpl_pc])



    annotations = os.path.join( Path( args.src_mesh_npy).parent , "curve_match.json")
    json_object = json.dumps(jdata, indent=4, cls=NumpyEncoder)
    #
    # print( json_object)
    # with open(annotations, "w") as outfile:
    #     outfile.write(json_object)