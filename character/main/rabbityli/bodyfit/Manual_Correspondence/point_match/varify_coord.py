import os.path


import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
import argparse
import pathlib
import open3d as o3d

DEPTH_SCALE = 1000


def depth_2_pc(depth, intrin):
    '''
    :param depth:
    :param intrin: 3x3 mat
    :return:
    '''

    fx, cx, fy, cy = intrin[0,0], intrin[0,2], intrin[1,1], intrin[1,2]
    cy = cx
    height, width = depth.shape

    u = np.arange(width) * np.ones([height, width])
    v = np.arange(height) * np.ones([width, height])

    v = np.transpose(v)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    mask = depth>0
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]

    return np.stack([X, Y, Z])



if __name__ == '__main__':



    # scan_dir = args.scan_dir
    scan_dir = "./data/smpl"
    src_id = 0 #args.src_id


    src_color = os.path.join( scan_dir,  "render_for_registration",  "color",   'cam-%04d'%src_id + ".png")
    src_color = cv2.imread( src_color )

    print ("src_color.shape" , src_color.shape)
    height , width , _= src_color.shape
    src_depth = os.path.join( scan_dir,  "render_for_registration",  "depth",   'cam-%04d'%src_id + ".png")
    src_depth = cv2.imread(src_depth, -1 ) / DEPTH_SCALE


    with open(os.path.join(scan_dir, "render_for_registration", "cam_parameters.json"), 'r') as f:
        cam_parameters = json.load(f)
        K =  np.array(cam_parameters['cam-%04d'%src_id]['k'])
        pose = np.array(cam_parameters['cam-%04d'%src_id]['pose'])

    pcd =  depth_2_pc(src_depth, K).reshape(3,-1).T

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pcd)
    pc.paint_uniform_color([1,0,0])
    pc.transform( pose )

    tex_mesh =  o3d.io.read_triangle_mesh( os.path.join(scan_dir, "manifold_full.obj"))
    tex_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries( [tex_mesh , pc])

