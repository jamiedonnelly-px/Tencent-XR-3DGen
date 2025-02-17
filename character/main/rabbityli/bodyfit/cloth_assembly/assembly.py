import os
import torch
import numpy as np

import cv2

def get_scale_ratio ( img ) :
    raw_h, raw_w, _ = ref_image.shape
    ratio_h, ratio_w = raw_h / 256. , raw_w / 256.
    ratio = ratio_h if ratio_h > ratio_w else ratio_w
    return ratio

with open('tripo/part.npy', 'rb') as f:
    full_point_cloud = np.load(f)
    full_valid_region = np.load(f)
    # full_point_cloud = full_point_cloud [ full_valid_region ]


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1, keepdims=True)
    data_zerocentered = data - data.mean(1, keepdims=True)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(dots / norms)

    print ("scale: %f " % s)

    trans = data.mean(1,keepdims=True) - s * rot * model.mean(1, keepdims=True)

    model_aligned = s * rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error, s

import open3d as o3d

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(full_point_cloud [ full_valid_region ])
pc.paint_uniform_color([1, 0, 0])
full_mesh = o3d.io.read_triangle_mesh( "tripo/mesh/full.obj")

o3d.visualization.draw([pc, full_mesh ])
ref_image = cv2.imread("tripo/img.png", -1)
ratio_full =  get_scale_ratio(ref_image)


def align_part( part_path = "/home/rabbityl/Dropbox/tripo/legs" ):
    # part_path = "/home/rabbityl/Dropbox/tripo/legs"

    import glob
    part_mesh = glob.glob(part_path + "/*/*.obj")[0]
    part_img = cv2.imread(os.path.join(part_path, "img.png" ), -1)

    with open(os.path.join(part_path, "part.npy"), 'rb') as f:
        part_pc = np.load(f)
        part_valid_region = np.load(f)


    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(part_pc[part_valid_region])
    pc.paint_uniform_color([1, 0, 0])
    part_mesh = o3d.io.read_triangle_mesh(part_mesh)
    # o3d.visualization.draw([pc, mesh])

    part_crop = np.loadtxt ( os.path.join(part_path, "crop.txt" ) )
    ratio = get_scale_ratio(part_img)

    crop =  (part_crop / ratio ).astype(int)

    full_vld_crop = full_valid_region[  crop[0][1]: crop[1][1] ,  crop[0][0]: crop[1][0] ]
    full_pcl_crop = full_point_cloud [   crop[0][1]: crop[1][1], crop[0][0]: crop[1][0]  ]



    # full_pcl_crop = full_pcl_crop[full_vld_crop]


    # scale_h = full_vld_crop.shape[0] /
    # scale_w = full_vld_crop.shape[1] / part_valid_region.shape[1]


    part_h, part_w = part_valid_region.shape[0], part_valid_region.shape[1]

    full_h, full_w = full_vld_crop.shape[0], full_vld_crop.shape[1]

    # xv = np.arange(part_w )
    # yu = np.arange(part_h )
    # v, u = np.meshgrid(xv, yu)

    resized_part_valid = cv2.resize(part_valid_region.astype(int), (full_w, full_h),  interpolation=cv2.INTER_NEAREST) > 0
    resized_part_pc = cv2.resize(part_pc, (full_w, full_h),  interpolation=cv2.INTER_NEAREST)
    mutual_vld =  np.logical_and( full_vld_crop, resized_part_valid  )



    match = [ full_pcl_crop[mutual_vld],  resized_part_pc[ mutual_vld ] ]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector( match[0])
    pc.paint_uniform_color([1, 0, 0])





    rot, trans, trans_error, s = align( match[1].T, match[0].T)

    match[1] = np.asarray( (s * rot * match[1].T + trans).T )

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(match[1])
    pc2.paint_uniform_color([0, 1, 0])

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector( match[0])
    pc.paint_uniform_color([1, 0, 0])



    part_verts = np.asarray( part_mesh.vertices )
    transformed_verts = np.asarray( (s * rot * part_verts.T + trans).T )
    part_mesh.vertices = o3d.utility.Vector3dVector(transformed_verts)
    part_mesh.paint_uniform_color([0.2, 0.9, 0.2])

    o3d.visualization.draw([pc, pc2, full_mesh, part_mesh])





    a =1



align_part()