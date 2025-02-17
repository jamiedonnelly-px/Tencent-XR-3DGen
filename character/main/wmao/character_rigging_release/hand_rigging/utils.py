import torch
from torch.nn import functional as F
import numpy as np
from torch import Tensor
from ipdb import set_trace as st
import copy

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    #print('°°°°° ',R.shape, t.shape)
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def lbs_customer(
    pose,
    v_posed,
    J,
    parents,
    lbs_weights,
    pose2rot: bool = True,
):
    # st()
    device, dtype = pose.device, pose.dtype
    batch_size = 1
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
    else:
        rot_mats = pose.reshape([batch_size,-1,3,3])
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # st()
    num_joints = len(parents)#.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    # st()
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    # st()
    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A, T


def batch_rigid_transform(
    rot_mats,
    joints,
    parents,
    dtype=torch.float32
):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    #print('--------------- ',rot_mats.shape, rel_joints.shape)

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, len(parents)):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def batch_rodrigues(
    rot_vecs,
    epsilon= 1e-8,
):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def lbs_customer_wscale(
    pose,
    v_posed,
    J,
    parents,
    lbs_weights,
    pose2rot: bool = False,
    log_scale = None
):
    # st()
    device, dtype = pose.device, pose.dtype
    if (len(pose.shape) == 3 and  not pose2rot) or len(pose.shape) == 2:
        batch_size = 1
    else:
        batch_size = pose.shape[0]
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
    else:
        rot_mats = pose.reshape([batch_size, -1, 3, 3])

    if log_scale is None:
        log_scale = torch.zeros([batch_size, rot_mats.shape[1], 3]).to(device=rot_mats.device,dtype=rot_mats.dtype)
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform_wscale(rot_mats, J, parents, dtype=dtype, log_scale=log_scale)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = len(parents)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)
    # st()
    v_posed = v_posed[None] #[1, nv, 3]
    homogen_coord = torch.ones([v_posed.shape[0], v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A, T


def batch_rigid_transform_wscale(
    rot_mats: Tensor,
    joints: Tensor,
    parents: Tensor,
    log_scale: Tensor,
    dtype=torch.float32,
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    # st()
    nj = len(parents)
    joints = joints.unsqueeze(0).unsqueeze(-1) #[b, jn, 3, 1]
    if rot_mats.shape[0] > 1:
        joints = joints.repeat([rot_mats.shape[0],1,1,1])

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    scaling_factors = torch.exp(log_scale)
    scaling_factors = scaling_factors.reshape(-1, nj, 3) #[batch, J, 3]
    scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1) #[batch, J, 3, 3]

    rot_mat_scale = [(rot_mats[:,0]@scale_factors_3x3[:,0])[:,None]]
    for i in range(1, nj):
        s_par_inv = torch.inverse(scale_factors_3x3[:, parents[i]])
        rot = rot_mats[:, i]
        s = scale_factors_3x3[:, i]
        rot_new = s_par_inv @ rot @ s
        rot_mat_scale.append(rot_new[:, None])
    rot_mats = torch.cat(rot_mat_scale,dim=1)

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    # st()
    transform_chain = [transforms_mat[:, 0]]

    for i in range(1, nj):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest

        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    # joints_homogen = torch.cat([joints,torch.ones_like(joints[:,:,:1])],dim=-1)[...,None]
    # st()
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms



part_colors = [
    [0.29803922, 0.44705882, 0.69019608],  # Blue
    [0.33333333, 0.65882353, 0.40784314],  # Green
    [0.76862745, 0.30588235, 0.32156863],  # Red
    [0.50588235, 0.44705882, 0.69803922],  # Purple
    [0.8       , 0.7254902 , 0.45490196],  # Brown
    [0.39215686, 0.70980392, 0.80392157],  # Cyan
    [0.6       , 0.6       , 0.6       ],  # Gray
    [1.        , 0.49803922, 0.        ],  # Orange
    [0.94117647, 0.89411765, 0.25882353],  # Yellow
    [0.8       , 0.4745098 , 0.65490196],  # Pink
    [0.7372549 , 0.74117647, 0.13333333],  # Olive
    [0.74117647, 0.33333333, 0.60392157],  # Magenta
    [0.09019608, 0.74509804, 0.81176471],  # Teal
    [0.98431373, 0.68627451, 0.89411765],  # Lavender
    [0.45098039, 0.83529412, 0.31372549],  # Lime
    [0.72156863, 0.5254902 , 0.04313725],  # Gold
    [0.70588235, 0.48627451, 0.78039216],  # Orchid
    [0.82745098, 0.7254902 , 0.8       ],  # PaleVioletRed
    [0.54117647, 0.74509804, 0.4745098 ],  # MediumSpringGreen
    [0.55294118, 0.82745098, 0.78039216],  # Aquamarine
    [0.52941176, 0.80784314, 0.98039216],  # LightSkyBlue
    [0.8       , 0.7254902 , 0.45490196],  # Peru
    [0.74509804, 0.68235294, 0.50196078],  # Tan
    [0.87843137, 0.54509804, 0.13333333],  # DarkOrange
    [0.37254902, 0.61960784, 0.62745098],  # CadetBlue
    [0.7372549 , 0.54117647, 0.55686275],  # IndianRed
    [0.78039216, 0.43137255, 0.88235294],  # MediumOrchid
    [0.82352941, 0.70588235, 0.54901961],  # PeachPuff
    [0.48235294, 0.82745098, 0.7372549 ],  # MediumAquamarine
    [0.55686275, 0.84705882, 0.62745098],  # MediumSeaGreen
    [0.74117647, 0.38823529, 0.54901961],  # PaleVioletRed
    [0.88235294, 0.43921569, 0.83921569],  # Orchid
    [0.68627451, 0.93333333, 0.93333333],  # LightCyan
    [0.95686275, 0.64313725, 0.37647059],  # DarkSalmon
    [0.67843137, 0.84705882, 0.90196078],  # LightSkyBlue
    [0.37254902, 0.61960784, 0.62745098],  # CadetBlue
    [0.7372549 , 0.54117647, 0.55686275]   # IndianRed
]

def interp_lbs_weights(points, template_points, lbs_weights, K=6, faces=None, faces_temp=None, weight=100,chamfer_weights=None):
    idxs, dists = knn(points, template_points, k=K, chamfer_weights=chamfer_weights)
    neighbs_weight = torch.exp(-weight*dists)
    neighbs_weight = neighbs_weight / neighbs_weight.sum(-1, keepdim=True)
    
    lbs_weights_merge = (lbs_weights[idxs] * neighbs_weight[:,:,None]).sum(dim=1) # K x J
    return lbs_weights_merge

def knn(query_points, points, k=3, chamfer_weights=None):
    """
    Find the k-nearest neighbors for each query point in the point cloud.

    Args:
    - query_points (torch.Tensor): Query points, shape (N, D), where N is the number of query points
      and D is the dimensionality of each point.
    - points (torch.Tensor): Point cloud, shape (M, D), where M is the number of points
      and D is the dimensionality of each point.
    - k (int): Number of neighbors to find.
    - chamfer_weights: NxM (0-1)

    Returns:
    - indices (torch.LongTensor): Indices of the k-nearest neighbors for each query point,
      shape (N, k).
    - distances (torch.Tensor): Euclidean distances to the k-nearest neighbors for each query point,
      shape (N, k).
    """
    squared_distances = torch.cdist(query_points, points, p=2) ** 2

    # st()
    if chamfer_weights is not None:
        squared_distances = squared_distances * chamfer_weights

    distances, indices = torch.topk(squared_distances, k=k, dim=1, largest=False)
    return indices, torch.sqrt(distances)

def weight_smoothing(weight, points, neighbors_dict, order=1, smoothing_type='mean', vids=None):
    # neighbors_dict = get_neighbour_from_faces(faces)
    weight_new = copy.deepcopy(weight)
    # st()
    if vids is None:
        vs = range(points.shape[0])
    else:
        vs = vids
    for i in vs:
        # neighbors = mesh.vertex_neighbors[i]
        neighbors = get_n_order_neighbour(neighbors_dict,i,order=order)
        
        if vids is not None:
            # st()
            neighbors = np.setdiff1d(neighbors, vids)
            order_tmp = order
            while len(neighbors) < 5:
                order_tmp = order_tmp + 1
                neighbors = get_n_order_neighbour(neighbors_dict,i,order=order_tmp)
                neighbors = np.setdiff1d(neighbors, vids)
        if len(neighbors) > 1:
            w = weight[neighbors]
            if smoothing_type == 'mean':
                # w = np.mean(np.concatenate([w,weight_new[i:i+1]]),axis=0)
                w = np.mean(w,axis=0)
            elif smoothing_type == 'median':
                w = np.median(w,axis=0)
            
            weight_new[i] = w
    return weight_new

def get_neighbour_from_faces(faces):
    neighbors = {}
    for i in range(faces.shape[0]):
        for j in faces[i]:
            if j in neighbors:
                neighbors[j] = neighbors[j].union(set(faces[i].tolist())-{j})
            else:
                neighbors[j]= set(faces[i].tolist())-{j}
    return neighbors

def get_n_order_neighbour(neighbors_dict,idx,order=1):
    # st()
    tmp_idx = [idx]
    his_idx = []
    for i in range(order):
        idx_new = []
        for id in tmp_idx:
            if id in his_idx:
                continue
            try:
                neig = neighbors_dict[id]
            except:
                st()
                print(id)
                continue
            idx_new += neig
            his_idx.append(id)
        idx_new = set(idx_new)
        tmp_idx = idx_new
    his_idx += list(tmp_idx)
    his_idx = list(set(his_idx) - {idx})
    return his_idx

def invert_lbs(lbs_weights, A, points):
    """
    lbs_weights: [N, num_joints]
    A: [bs,num_joints,4,4]
    points:[N,3]
    """
    batch_size, num_joints, _, _ = A.shape
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    transform_merge = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)
    homogen_coord = torch.ones([points.shape[0], 1],
                            dtype=A.dtype, device=A.device)
    v_posed_homo = torch.cat([points, homogen_coord], dim=1)
    v_homo = torch.matmul(transform_merge[0].inverse(), torch.unsqueeze(v_posed_homo, dim=-1))
    verts_cano = v_homo[:, :3, 0]
    return verts_cano, transform_merge


def get_edges_from_faces(faces):
    # Initialize a set to store unique edges
    edges = set()

    # Iterate through all triangles
    for triangle in faces:
        # Extract vertices of the triangle
        v0, v1, v2 = triangle
        
        # Form edges from vertices
        edge1 = tuple(sorted((v0, v1)))
        edge2 = tuple(sorted((v1, v2)))
        edge3 = tuple(sorted((v2, v0)))
        
        # Add edges to the set
        edges.update([edge1, edge2, edge3])

    edges = np.array([list(e) for e in edges])
    return edges


def get_edges_from_neighbors(neighbors):
    # Initialize a set to store unique edges
    edges = set()

    # Iterate through all triangles
    for k, v in neighbors.items():
        # Extract vertices of the triangle
        
        # Form edges from vertices
        for vv in v: 
            edge = tuple(sorted((k, vv)))
            # Add edges to the set
            edges.update([edge])

    edges = np.array([list(e) for e in edges])
    return edges

@torch.no_grad()
def add_edge_with_thre(verts, device, thres=0.001):
    verts = torch.from_numpy(verts).to(device=device,dtype=torch.float16)
    edges_new = set()
    for i in range(verts.shape[0]-2):
        dist = torch.norm(verts[i:i+1] - verts[i+1:],dim=-1)
        idx = torch.where(dist<thres)[0].cpu().data.numpy().tolist()
        if len(idx) > 0:
            edge_tmp = []
            for j in idx:
                edge_tmp.append(tuple(sorted((i, j+i+1))))
            edges_new.update(edge_tmp)
    edges_new = np.array([list(e) for e in edges_new])
    same_v = []
    tmp = dict()
    for i in np.arange(edges_new.shape[0]):
        # try:
        e = edges_new[i]
        if e[0] in tmp.keys():
            if e[0] not in same_v[tmp[e[0]]]:
                same_v[tmp[e[0]]].append(e[0])
            if e[1] not in same_v[tmp[e[0]]]:
                same_v[tmp[e[0]]].append(e[1])
            tmp[e[1]] = tmp[e[0]]
        elif e[1] in tmp.keys():
            if e[0] not in same_v[tmp[e[1]]]:
                same_v[tmp[e[1]]].append(e[0])
            if e[1] not in same_v[tmp[e[1]]]:
                same_v[tmp[e[1]]].append(e[1])
            tmp[e[0]] = tmp[e[1]]
        else:
            tmp[e[0]] = tmp[e[1]] = len(same_v)
            same_v.append([e[0],e[1]])
        # except:
        #     print(1)
        #     st()
    return edges_new, same_v


@torch.no_grad()
def repeat_point_consist(points, weights):
    # st()
    thres = 0.001
    cdist = torch.cdist(points, points)
    cdist = cdist + torch.eye(cdist.shape[0], device=cdist.device,dtype=cdist.dtype)
    idxs, _ = torch.nonzero(cdist<thres,as_tuple = True)
    idxs = list(set(idxs.cpu().data.numpy().tolist()))
    weight_new = weights.clone()
    for i in idxs:
       dist = torch.norm(points[i:i+1] - points,dim=-1)
       idx = torch.where(dist<thres)[0]
       weight_new[i] = weights[idx].mean(dim=0,keepdim=True)
    return weight_new


@torch.no_grad()
def copy_weight_to_repeated_point(weights, same_vs, vids_mask=None):
    
    weights_new = weights.detach().clone()
    for vs in same_vs:
        if vids_mask is not None:
            vs_valid = list(set(vs) - set(vids_mask))
        else:
            vs_valid = vs
        weights_new[vs] = weights[vs_valid].mean(dim=0,keepdim=True)
    return weights_new