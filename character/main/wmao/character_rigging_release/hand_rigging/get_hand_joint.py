import argparse
import sys
import os
from models.point_transformer_partseg import *
import trimesh
from ipdb import set_trace as st
import numpy as np
import scipy.linalg
import open3d as o3d
from utils import *
from train_point_transformer_contrast_new import load_data

def remove_noise_points(pts, threshold):
    cdist = np.linalg.norm(pts[None]-pts[:,None],axis=-1)
    cdist = cdist + np.eye(cdist.shape[0]) * 10000

    adj = (cdist < threshold)*1.0
    deg = np.diag(adj.sum(axis=0))

    laplacian = deg - adj

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(laplacian)
    num_connected_components = np.sum(np.isclose(eigenvalues, 0, atol=1e-10))
    
    # Identify eigenvectors corresponding to zero eigenvalues
    zero_eigenvector_indices = np.where(np.isclose(eigenvalues, 0, atol=1e-10))[0]
    zero_eigenvectors = eigenvectors[:, zero_eigenvector_indices]
    # Group nodes into connected components

    n_nodes = pts.shape[0]
    connected_components = []
    num_per_comp = []
    visited = set()

    for i in range(n_nodes):
        if i not in visited:
            # Find nodes with similar eigenvector entries
            component = set()
            for j in range(n_nodes):
                if np.allclose(zero_eigenvectors[i], zero_eigenvectors[j], atol=1e-10):
                    component.add(j)
                    visited.add(j)
            num_per_comp.append(len(component))
            connected_components.append(component)
    # for pi, comp in enumerate(connected_components):
    #     if len(comp) > 50:
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(pts[list(comp)])
    #         o3d.io.write_point_cloud(f'./{args.out_dir}/{pi:04d}.ply', pcd)
    num_per_comp = np.array(num_per_comp)
    idx = np.argmax(num_per_comp)
    max_comp_idx = list(connected_components[idx])
    # if len(comp_idx) > 
    # tmp = 1
    # for comp in connected_components:
    #     if len(comp) > tmp:
    #         tmp = len(comp)
    #         max_connected_idxs = list(comp)

    return pts[max_comp_idx]

def point2line_dist(pts, p0, p1):
    """_summary_
    compute the distance from a set of point to a line (not a segment)
    Args:
        pts: [n x 3]
        p0: [None x 3] start of the line
        p1: [n x 3] end of the line
    """     
    v1 = pts[:,None] - p0[None] # m x none x 3
    v2 = p1 - p0 #[n x 3]
    v2 = v2 / (np.linalg.norm(v2,axis=-1)[:,None] + 1e-5) # n x 3
    dist = (v1 * v2[None]).sum(axis=-1) # m x n
    return dist


def find_rotation_matrix_between_vectors(v1, v2):
    # Normalize the input vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate the cross product and dot product
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)

    # Skew-symmetric matrix for the cross product
    skew_symmetric_matrix = np.array([
        [0, -cross_product[2], cross_product[1]],
        [cross_product[2], 0, -cross_product[0]],
        [-cross_product[1], cross_product[0], 0]
    ])

    # Rotation matrix using Rodrigues' formula
    rotation_matrix = np.eye(3) + skew_symmetric_matrix + np.dot(skew_symmetric_matrix, skew_symmetric_matrix) * (1 - dot_product) / (np.linalg.norm(cross_product) ** 2)

    return rotation_matrix

def draw_joints(joints, parents, pos_fix='', out_dir='./'):
    """_summary_

    Args:
        joints (_type_): a list of joints with None value means no joint position available
        parents (_type_): _description_
        pos_fix (str, optional): _description_. Defaults to ''.
        out_dir (str, optional): _description_. Defaults to './'.
    """
    # visualize the skeleton and get the index
    mesh = o3d.geometry.TriangleMesh()
    fn = 0
    verts = []
    faces = []
    rad = 0.01
    # st()
    for ji in range(len(joints)):
        if len(joints[ji]) == 0:
            continue
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=rad,resolution=5)
        mesh.translate(joints[ji])
        # o3d.io.write_triangle_mesh(f'./joints_{ji:02d}.obj', mesh)
        verts.append(np.array(mesh.vertices))
        faces.append(np.array(mesh.triangles)+fn)
        fn += verts[-1].shape[0]
        if parents[ji] >= 0:
            point1 = joints[parents[ji]]
            point2 = joints[ji]
            direction = point2 - point1
            arrow_length = np.linalg.norm(direction)
            if arrow_length <= 0:
                print(parents[ji],ji)
                continue
            # print(arrow_length)
            # Create arrow geometry
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=rad/5,  # Adjust the thickness of the arrow
                cone_radius=rad/4,       # Adjust the size of the arrow head
                cylinder_height=arrow_length*0.8,   # Adjust the length of the arrow
                cone_height=arrow_length * 0.2
            )
            # Calculate rotation matrix to align arrow with direction vector
            rotation = find_rotation_matrix_between_vectors(np.array([0,0,1.]),direction)
            arrow.rotate(rotation, center=(0, 0, 0))
            # Translate arrow to the starting point
            arrow.translate(point1)
            verts.append(np.array(arrow.vertices))
            faces.append(np.array(arrow.triangles)+fn)
            fn += verts[-1].shape[0]
    verts = np.concatenate(verts,axis=0)
    faces = np.concatenate(faces,axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(f'{out_dir}_skeleton{pos_fix}.obj', mesh)

def compute_principal_directions(points):
    """
    Computes the principal directions of a point cloud using PCA.

    Parameters:
    points (numpy.ndarray): A Nx3 array representing the 3D point cloud.

    Returns:
    eigenvalues (numpy.ndarray): Eigenvalues indicating variance along each principal direction.
    eigenvectors (numpy.ndarray): Eigenvectors representing the principal directions.
    """
    # Center the point cloud by subtracting the mean
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_points.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    return eigenvalues, eigenvectors

def main(model, model_partseg, input_mesh, npts, out_dir, pts):
    """_summary_

    Args:
        model: _description_
        model_partseg: _description_
        input_mesh: _description_
        npts: _description_
        args: _description_
    """
    part_col = np.array(part_colors)
    # mesh = trimesh.load(input_mesh)
    # try:
    #     pts, face_indices = trimesh.sample.sample_surface(mesh, npts)
    # except:
    #     print('sample point error')
    # verts = mesh.vertices
    # faces = mesh.faces
    # vmean = verts.mean(axis=0)
    # verts = verts - vmean
    # vmax = verts.max(axis=0)
    # vmin = verts.min(axis=0)
    # scale = 2 / np.linalg.norm(vmax - vmin)
    # pts = (pts - vmean) * scale
    
    pts = pts[np.random.choice(np.arange(pts.shape[0]),npts)]

    points = torch.from_numpy(pts).float().cuda()
    n = points.shape[0]
    inp = {
        "coord": points.reshape(n, 3),
        "feat": points.reshape(n, -1),
        "offset": torch.cumsum(torch.ones(1).cuda()*n,dim=0)
    }
    seg_pred = model(inp) # [b*2*n, out_feat]
    part_logit = model_partseg(seg_pred)
    parts = part_logit.max(dim=-1)[1].cpu().data.numpy()
    part_names = ['hand','thumb','index','middle','ring','pinky']


    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(pts)
    mesh.colors = o3d.utility.Vector3dVector(part_col[parts])
    o3d.io.write_point_cloud(f'{out_dir}_parts.ply', mesh)

    joints = {}
    threshold = 0.07
    pts_hand = pts[parts==0]
    pts_hand = remove_noise_points(pts_hand, threshold)
    hand = pts_hand.mean(axis=0)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts_hand)
    # col = np.zeros_like(pts_hand)
    # col[:] = part_col[:1]
    # pcd.colors = o3d.utility.Vector3dVector(col)
    # o3d.io.write_point_cloud(f'{out_dir}/{part_names[0]}.ply', pcd)
    for i in range(1, 6):
        idxs = np.where(parts==i)[0]
        pts_part = pts[idxs]
        # pts_part = np.concatenate([pts[parts==1],pts[[parts==2]]],axis=0)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts_part)
        # col = np.zeros_like(pts_part)
        # col[:] = part_col[i:i+1]
        # pcd.colors = o3d.utility.Vector3dVector(col)
        # o3d.io.write_point_cloud(f'{out_dir}/{part_names[i]}.ply', pcd)
        
        if pts_part.shape[0] < 150:
            continue
        try:
            pts_part = remove_noise_points(pts_part, threshold)
        except:
            st()
        if pts_part.shape[0] < 150:
            continue
        
        eigenvalues, eigenvectors = compute_principal_directions(pts_part)
        pca = eigenvectors[:,0]
        
        fing_mean = pts_part.mean(axis=0)
        hand2fing = fing_mean - hand
        if (pca * hand2fing).sum() < 0:
            pca = -pca
        
        # root joint
        proj_coord = ((pts_part - fing_mean) * pca[None]).sum(axis=-1)
        jroot = pca * np.min(proj_coord) + fing_mean
        jend = pca * np.max(proj_coord) + fing_mean

        if False:
            arrow_length = 0.5
            rad = 0.01
            # draw direction
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=rad/5,  # Adjust the thickness of the arrow
                    cone_radius=rad/4,       # Adjust the size of the arrow head
                    cylinder_height=arrow_length*0.8,   # Adjust the length of the arrow
                    cone_height=arrow_length * 0.2
                )
            # Calculate rotation matrix to align arrow with direction vector
            rotation = find_rotation_matrix_between_vectors(np.array([0,0,1.]),pca)
            arrow.rotate(rotation, center=(0, 0, 0))
            # Translate arrow to the starting point
            arrow.translate(pts_part.mean(axis=0))
            o3d.io.write_triangle_mesh(f'{out_dir}/{part_names[i]}_dir.ply', arrow)


        # # get hand and finger connection
        # cdist = np.linalg.norm(pts_part[None] - pts_hand[:,None],axis=-1)
        # thre = 0.06
        # for _ in range(3):
        #     idxs1, idxs2 = np.where(cdist < thre)
        #     idxs1 = np.unique(idxs1)
        #     idxs2 = np.unique(idxs2)
        #     if (len(idxs1) + len(idxs2)) < 10:
        #         thre = thre * 1.1
        # pts_conn = np.concatenate([pts_part[idxs2], pts_hand[idxs1]],axis=0)
        # jroot = pts_conn.mean(axis=0)
        
        # dist_p2l = point2line_dist(pts_part, jroot[None], pts_part[::2])
        # dist_std = np.std(dist_p2l,axis=0)
        # idx = np.argmin(dist_std)
        # jend = pts_part[idx] 
        j1 = (jend - jroot)/3 + jroot
        j2 = (jend-jroot)*2/3 + jroot
        # if pts_part.shape[0] < 100:
        #     continue
        joints[part_names[i]] = np.concatenate([jroot[None], j1[None], j2[None], jend[None]],axis=0)
    
    # draw joint
    joints_all = hand[None]
    joints_name = ['hand']
    parents = [-1]
    for k, v in joints.items():
        jn = joints_all.shape[0]
        parents.extend([0,jn,jn+1,jn+2])
        joints_name.extend([k]*4)
        joints_all = np.concatenate([joints_all, v],axis=0)
        
    draw_joints(joints=joints_all, parents=parents, out_dir=out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model')

    # model config
    parser.add_argument('--model', type=str, default='PointTransformerSeg38', help='choose from: PointTransformerSeg26, PointTransformerSeg38, PointTransformerSeg50') # 
    parser.add_argument('--model_ckpt', type=str, default='output/hand_part_seg_rot_aug_PointTransformerSeg38/2024-12-10_14-46-43/checkpoints/best_model.pth', help='consume training')
    parser.add_argument('--npts', type=int, default=2048, help='consume training')
    parser.add_argument('--input_dir', type=str, default='data/vis/', help='dataset dir')
    parser.add_argument('--input_mesh', type=str, default='data/vis/Ch19_nonPBR_right.ply', help='dataset dir')
    parser.add_argument('--out_dir', type=str, default='output/rigging', help='dataset dir')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    in_channels = 3
    out_feat = 128
    n_parts = 6
    model = PointTransformerSeg38(in_channels=in_channels, num_classes=out_feat).float().cuda() 
    model_partseg = torch.nn.Linear(in_features=out_feat, out_features=n_parts).float().cuda()
    model.eval()
    model_partseg.eval()

    checkpoint = torch.load(args.model_ckpt)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model_partseg.load_state_dict(checkpoint['model_partseg_state_dict'])
    print(f'load ckpt at epoch {start_epoch:04d}')
    pts, parts, file_names = load_data('/aigc_cfs_2/weimao/hand_rigging/data/npz', is_augment=False, mode='test')

    # for mesh_name in os.listdir(args.input_dir):
    #     if not mesh_name.endswith('.ply'):
    #         continue
    for i, mesh_name in enumerate(file_names):
        print(f'processing {mesh_name}')
        input_mesh = os.path.join(args.input_dir,mesh_name)
        out_dir = os.path.join(args.out_dir, mesh_name.split('.ply')[0])
        # os.makedirs(out_dir, exist_ok=True)
        main(model, model_partseg, input_mesh, args.npts, str(out_dir)[:-1], pts[i])
    

        
