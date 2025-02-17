import os.path


import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
import argparse
import pathlib
import open3d as o3d
import trimesh

DEPTH_SCALE = 1000

from matplotlib import colors
import matplotlib.colors as mcolors
template_colors = [ colors.to_rgba(key)[:-1] for key in mcolors.TABLEAU_COLORS ]
clst = []
for  i in range (30): # assume max 300 points
    clst = clst + template_colors


def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def sample_barycentric_coordinates(vertices, faces, n_samples=10000):
    """
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    Parameters:
    vertices  - n x 3 matrix
    faces     - n x 3 matrix
    n_samples - positive integer

    Return:
    vertices - point cloud

    Reference :
    [1] Barycentric coordinate system

    \begin{align}
      P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
    \end{align}
    """
    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)
    # Sample exactly n_samples. First, oversample points and remove redundant
    # Error fix by Yangyan (yangyan.lee@gmail.com) 2017-Aug-7
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1
    n_samples = np.sum(n_samples_per_face)
    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample
    r = np.random.rand(n_samples, 2);
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    wa = (1 - np.sqrt(r[:,0:1]))
    wb = np.sqrt(r[:,0:1]) * (1 - r[:,1:])
    wc = np.sqrt(r[:,0:1]) * r[:,1:]
    P = wa * A + wb * B + wc * C  # point position
    anchors = faces[sample_face_idx] # anchor vertices
    w = np.concatenate([ wa, wb, wc], axis=-1) # barycentric corrdinates
    return P, w, anchors

def knn_point_np(k, reference_pts, query_pts):
    '''
    :param k: number of k in k-nn search
    :param reference_pts: (N, 3) float32 array, input points
    :param query_pts: (M, 3) float32 array, query points
    :return:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''

    N, _ = reference_pts.shape
    M, _ = query_pts.shape
    reference_pts = reference_pts.reshape(1, N, -1).repeat(M, axis=0)
    query_pts = query_pts.reshape(M, 1, -1).repeat(N, axis=1)
    dist = np.sum((reference_pts - query_pts) ** 2, -1)
    idx = partition_arg_topK(dist, K=k, axis=1)
    val = np.take_along_axis ( dist , idx, axis=1)
    return np.sqrt(val), idx

def partition_arg_topK(matrix, K, axis=0):
    """ find index of K smallest entries along a axis
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def depth_2_pc(depth, intrin):
    '''
    :param depth:
    :param intrin: 3x3 mat
    :return: point cloud image reshaped to [H*W, 3]
    '''

    fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    height, width = depth.shape
    u = np.arange(width) * np.ones([height, width])
    v = np.arange(height) * np.ones([width, height])
    v = np.transpose(v)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return np.stack([X, Y, Z])

def read_rgb_pcd(src_scan_dir, src_id):
    src_color = os.path.join( src_scan_dir,   "render_for_registration/color",   'cam-%04d'%src_id + ".png")



    src_color = cv2.imread( src_color )
    src_depth = os.path.join( src_scan_dir,   "render_for_registration/depth",   'cam-%04d'%src_id + ".png")

    print( src_depth)
    src_depth = cv2.imread(src_depth, -1 )  / DEPTH_SCALE
    src_depth_vis = src_depth * DEPTH_SCALE * 255 / 65535

    with open(os.path.join(src_scan_dir, "render_for_registration", "cam_parameters.json"), 'r') as f:
        cam_parameters = json.load(f)
        K =  np.array(cam_parameters['cam-%04d'%src_id]['k'])
        src_pose = np.array(cam_parameters['cam-%04d'%src_id]['pose'])
    src_pcd_im =  depth_2_pc(src_depth, K).reshape(3,-1).T

    return src_color, src_pcd_im, src_pose, src_depth_vis


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sdir', type=str, default='')
    parser.add_argument('--tdir', type=str, default='/home/rabbityl/workspace/auto_rig/bodyfit/Manual_Correspondence/data/smpl_mesh_1280/smpl')
    parser.add_argument('--id', type=int, default=0)

    parser.add_argument('--reuse', action='store_true')

    # parser.add_argument('--tid', type=int, default=0)
    args = parser.parse_args()

    src_scan_dir = args.sdir  #"./data/garen"  #path to object scan
    tgt_scan_dir = args.tdir  #"./data/garen"  #path to object scan
    src_id = args.id  # frame id
    tgt_id = args.id  # frame id


    annotations = os.path.join( src_scan_dir, "correspondence" ) # matches_" + '%0d'%src_id + "_" + '%06d'%tgt_id + ".json"  )
    pathlib.Path( annotations ).mkdir(exist_ok=True)
    annotations = os.path.join( annotations ,  "obj2smpl_" + '%04d' % src_id + "_" + '%04d' % tgt_id + ".json" )


    ## load src and target
    src_color, src_pcd_im, src_pose, src_depth = read_rgb_pcd(src_scan_dir, src_id )
    tgt_color, tgt_pcd_im, tgt_pose, tgt_depth = read_rgb_pcd(tgt_scan_dir, tgt_id )
    height , width , _= src_color.shape

    

    colors = np.concatenate ( [src_color , tgt_color ],axis=1)
    depths = np.concatenate ( [src_depth, tgt_depth] , axis= 1)

    current_corr= colors.copy()

    correspondence= colors.copy()


    if args.reuse:
        if os.path.exists(annotations) :
            reuse_anno = load_json(annotations)
            pairs = reuse_anno["pairs"]

    pairs = []
    # color = []
    current_pair =  [None, None]

    def click(event, x, y, flags, param):
        # grab references to the global variables
        # global refPt, cropping
        global  status, depths, colors, pairs, width, height , current_pair
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < width :
                current_pair[0] = [x,y]
            else :
                current_pair[1] = [ x-width, y ]
            print (current_pair)



    cv2.namedWindow("correspondence" , 0 )
    cv2.resizeWindow("correspondence", width*2, height)
    cv2.setMouseCallback("correspondence", click)

    # cv2.setMouseCallback('depth', click)

    # cv2.namedWindow( "selector" , 0 )
    # cv2.resizeWindow("selector", width*2, height)
    # cv2.setMouseCallback("selector", click)


    while True :

        if current_pair[0] != None :
            cv2.circle(correspondence, (current_pair[0][0], current_pair[0][1]), 3, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(correspondence, (current_pair[0][0], current_pair[0][1]), 4, (20,20,20), 1, cv2.LINE_AA)

        if  current_pair[1] != None:
            cv2.circle(correspondence, (current_pair[1][0] + width, current_pair[1][1]), 3, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(correspondence, (current_pair[1][0] + width, current_pair[1][1]), 4, (20,20,20), 1, cv2.LINE_AA)
            # cv2.circle(correspondence, (current_pair[1][0], current_pair[1][1]), 4, (20, 20, 20), 1, cv2.LINE_AA)

        if current_pair[0] != None and current_pair[1] != None:
            cv2.line(correspondence,  (current_pair[0][0], current_pair[0][1]), (current_pair[1][0]+width, current_pair[1][1]), (255,255,0),1 , cv2.LINE_AA )


        for cid, ele in enumerate ( pairs ) :
            # print ele
            x2 = ele[1][0] + width
            y2 = ele[1][1]
            c = clst[cid]
            c = tuple ( [ int(255 * x) for x in c] )
            # print( "c:", c)
            cv2.circle(correspondence, (ele[0][0],ele[0][1]),3,  c, -1, cv2.LINE_AA)
            cv2.circle(correspondence, (ele[0][0],ele[0][1]),4,  (20,20,20), 1, cv2.LINE_AA)
            cv2.circle(correspondence, (x2, y2), 3, c,   -1,cv2.LINE_AA)
            cv2.circle(correspondence, (x2, y2), 4, (20,20,20),   1,cv2.LINE_AA)
            # cv2.line(correspondence, (ele[0][0],ele[0][1]), (x2, y2),  (255,255,0),   1 , cv2.LINE_AA )

        cv2.imshow("correspondence", correspondence)

        # cv2.imshow("selector",  current_corr)

        correspondence = colors.copy()

        current_corr = colors.copy()

        # print pairs
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):

            if current_pair[0] != None  and current_pair[1] != None :
                aPair = copy.deepcopy(current_pair)
                pairs.append(aPair)

                current_pair = [None, None]
                print ("pairs", pairs)

        elif key == ord('b'):# Roll back one correspondence (to correct mis-click)
            if len(pairs) > 0 :
                pairs = pairs[:-1]
                print("pairs", pairs)
        elif key == ord('d'):
            break

    cv2.destroyAllWindows()



    # transform selected pixels to 3D world coordinate
    pairs = np.asarray(pairs).astype(int)
    src_uv, tgt_uv = pairs[:,0], pairs[:,1]
    src_idx = src_uv[:,1] * width + src_uv[:,0]
    tgt_idx = tgt_uv[:,1] * width + tgt_uv[:,0]
    src_pts = src_pcd_im[src_idx]
    tgt_pts = tgt_pcd_im[tgt_idx]
    src_pts = ( src_pose[:3,:3] @ src_pts.T + src_pose[:3,3:] ).T
    tgt_pts = ( tgt_pose[:3,:3] @ tgt_pts.T + tgt_pose[:3,3:] ).T

    tgt_pcd_im = ( tgt_pose[:3,:3] @ tgt_pcd_im.T + tgt_pose[:3,3:] ).T





    #load SMPL mesh
    smpl_faces = np.load( os.path.join( tgt_scan_dir, "smpl_faces.npy")  )
    smpl_verts = np.load( os.path.join( tgt_scan_dir, "smpl_verts.npy")  )
    smplmesh = o3d.geometry.TriangleMesh()
    smplmesh.vertices = o3d.utility.Vector3dVector(smpl_verts)
    smplmesh.triangles = o3d.utility.Vector3iVector(smpl_faces)
    smplmesh.compute_vertex_normals()



    # compute barycentric coordinates on SMPL surface
    # smpl_points, baryc_coords, verts_id = sample_barycentric_coordinates(np.asarray(smplmesh.vertices), np.asarray(smplmesh.triangles))
    # dists, idx = knn_point_np(1, smpl_points, tgt_pts)
    # valid_mask = dists.squeeze() < 0.1
    # idx = idx [valid_mask].squeeze()
    # src_pts = src_pts[valid_mask]
    # baryc_coords1, verts_id1 =  baryc_coords[idx], verts_id[idx]

    # cast ray to obtain barycentric coordinates on SMPL surface
    trimesh_smpl = trimesh.Trimesh(smpl_verts , smpl_faces)
    ray_origin = tgt_pose[:3, 3:].squeeze()
    mask = []
    baryc_coords = []
    verts_id = []
    for i in range(len(tgt_pts)):
        ray_direction = tgt_pts[i] - ray_origin
        ray_direction = ray_direction / np.linalg.norm( ray_direction)
        locations, index_ray, index_tri = trimesh_smpl.ray.intersects_location( ray_origins=ray_origin[None], ray_directions=ray_direction[None])
        if len(locations)<1:
            mask.append(False)
            continue
        dist =  np.linalg.norm( locations - tgt_pts[i][None], axis=1 )
        ind = np.argmin(dist)
        if dist[ind] < 0.05:
            mask.append(True)
            vid = smpl_faces [ index_tri[ind]]
            _3_verts = smpl_verts [ vid ]
            baryc = np.linalg.inv( _3_verts.T) @ locations[ind]
            baryc_coords.append( baryc )
            verts_id.append( vid)
        else:
            mask.append(False)

    baryc_coords = np.stack( baryc_coords, axis=0 )
    verts_id = np.stack( verts_id, axis=0 )
    src_pts = src_pts [mask]


    ### apply mirror operation to double correspondence
    with open ("smpl_left_right_symmetric_map.npy", "rb") as f :
        mirror_vert_map = np.load(f)


    mirror_verts_id =  mirror_vert_map[1][verts_id.reshape(-1)].reshape(-1,3)
    mirror_baryc_coords = baryc_coords.copy()
    mirror_src_pts = src_pts.copy()
    mirror_src_pts[:,0] = mirror_src_pts[:,0]  * -1

    verts_id = np.concatenate([ verts_id, mirror_verts_id ] , axis=0 )
    baryc_coords = np.concatenate([ baryc_coords, mirror_baryc_coords ] , axis=0 )
    src_pts = np.concatenate([ src_pts, mirror_src_pts ] , axis=0 )



    viz=True
    if viz:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(smplmesh.vertices)[verts_id.reshape(-1)] )
        pc.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries( [ smplmesh, pc ])

        pc1 = o3d.geometry.PointCloud()
        pc1.points = o3d.utility.Vector3dVector( src_pts )
        pc1.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries( [smplmesh, pc, pc1 ])



    # save correspondence
    json_str = {
        "character_pts": src_pts,
        "smpl_baryc_coords": baryc_coords,
        "smpl_verts_id": verts_id,
        "pairs": pairs
    }
    json_object = json.dumps(json_str, indent=4, cls=NumpyEncoder)
    with open( annotations, "w") as outfile:
        outfile.write(json_object)


