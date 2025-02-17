import bpy
import os
from pathlib import Path
import string
import glob
import numpy as np
import bmesh
import time

import bpy
import bmesh
from mathutils.bvhtree import BVHTree
import numpy as np

from numpy import arange, pi, sin, cos, arccos
from mathutils import Matrix, Vector, Quaternion, Euler
import math


valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)

import sys
# sys.path.insert(0, "..")
# from utils import *



def put_cam_around_obj( n_cam, obj_center, length, fix_kpts = None):

    pi = 3.14

    def sphere_point_sample(n=300):
        # use fibonacci spiral
        goldenRatio = (1 + 5 ** 0.5) / 2
        i = arange(0, n)
        theta = 2 * pi * i / goldenRatio
        phi = arccos(1 - 2 * (i + 0.5) / n)
        x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
        return np.stack([x, y, z], axis=-1)

    def XZ_plane_sample (n=10):

        theta = 2*pi *  arange(0, n) / n
        x = sin(theta)
        z = cos(theta)
        y = np.array([0]*n)
        return   np.stack([x, y, z], axis=-1)


    def look_at(obj_camera, point):
        loc_camera = obj_camera.location
        direction = point - loc_camera
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        # assume we're using euler rotation
        obj_camera.rotation_euler = rot_quat.to_euler()

    def set_camera(bpy_cam, angle=pi / 3, W=600, H=600):
        bpy_cam.angle = angle
        bpy_scene = bpy.context.scene
        bpy_scene.render.resolution_x = W
        bpy_scene.render.resolution_y = H

    #设置Camera，在物体为中心的球面上采样相机
    points = XZ_plane_sample(n_cam)
    points = points * length * 3  #scale
    points = obj_center[None] + points
    cam_names = ["cam-%04d" % i for i in range (n_cam)]
    for i in range (n_cam):
        camera_data = bpy.data.cameras.new(name=cam_names[i])
        camera_object = bpy.data.objects.new(cam_names[i], camera_data)
        bpy.context.scene.collection.objects.link(camera_object)
        camera_object.location = Vector (points[i])
        look_at_point =  Vector(obj_center) #Vector((0,0,0))
        camera_data.display_size = 0.02
        camera_data.clip_start = 0.01
        camera_data.clip_end = 100
        set_camera(camera_data, angle=pi/3, W=600, H=600)
        look_at(camera_object, look_at_point)
        bpy.context.view_layer.update() #update camera params


    # add camera around key points
    if fix_kpts is not None:
        for k, v in fix_kpts.items():
            cam_name = "cam-" + k
            cam_names.append( cam_name )
            camera_data = bpy.data.cameras.new(name=cam_name)
            camera_object = bpy.data.objects.new(cam_name, camera_data)
            bpy.context.scene.collection.objects.link(camera_object)
            camera_object.location = Vector( (v[0],v[1],v[2]) )
            camera_data.display_size = 0.01
            camera_data.clip_start = 0.01
            camera_data.clip_end = 100
            bpy.context.view_layer.update()
    return [ bpy.data.objects[ cam] for cam in cam_names]




def compute_mesh_size( meshes ,skip=1) :

    #计算角色中心点,身体长度
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    verts = []
    for ind, mesh in enumerate(meshes):
        print(ind)
        vert, _ = read_mesh_to_ndarray( mesh, mode="edit", skip=skip)
        mat = np.asarray(mesh.matrix_world)
        R,t = mat[:3,:3], mat[:3,3:] #Apply World Scale
        verts.append( ( R@vert.T + t ).T )
    verts=np.concatenate(verts, axis=0)
    print("verts", verts.shape, verts.min(axis=0), verts.max(axis=0))
    min_ = verts.min(axis=0)
    max_ = verts.max(axis=0)
    obj_center = (min_ + max_) / 2
    length = np.linalg.norm( max_ - min_ )/2
    print( "edit mode ", obj_center, length)

    return obj_center, length

def read_mesh_to_ndarray( mesh, mode = "Edit", skip=1):
    ''' read the vert coordinate of a deformed mesh
    :param mesh: mesh object
    :return: numpy array of the mesh
    '''
    assert mode in [ "edit", "object"]

    if mode == "object" :
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(mesh, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray( mverts_co, dtype=np.float32)
        # faces = [[ v.index for v in face.verts ] for face in bm.faces]
        # faces=np.asarray(faces,dtype=np.int32)
        bm.free()
    elif mode == "edit" :
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        if mesh.mode != "EDIT" :
            bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        bm.verts.ensure_lookup_table()
        mverts_co = []
        for i in range(0, len(bm.verts), skip):
            mverts_co.append( bm.verts[i].co )
        # mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray( mverts_co, dtype=np.float32)
        # faces = [[ v.index for v in face.verts ] for face in bm.faces]
        # faces=np.asarray(faces,dtype=np.int32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co, None



def delete_occluded_faces_from_cameras(themesh, cameras, fix_normal=False,   anchors = [ [0.333, 0.333, 0.334] ]):
    '''
    :param themesh:
    :param cameras:
    :return:
    '''

    # return

    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = themesh
    bpy.ops.object.mode_set(mode='EDIT')
    me = themesh.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()


    raycast_mesh = themesh
    # construct camera tuples
    camera_items = []
    for cam in cameras:
        ray_origin = cam.location
        ray_begin_local = raycast_mesh.matrix_world.inverted() @ ray_origin
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bvhtree = BVHTree.FromObject(raycast_mesh, depsgraph)
        # bvhtree = BVHTree.FromBMesh(bm)
        item = ( ray_origin, ray_begin_local, bvhtree)
        camera_items.append( item )
    assert themesh.type == "MESH"


    faces_select = [] # list of faces to retain
    faces_flip = [] # list of faces that has flip normal

    for idx, face in enumerate( bm.faces) :
        # if idx%1000==0 :
        # print( idx,"/", len(bm.faces))
        observed = False
        vert_vector = [v.co for v in face.verts ]

        for anchor in anchors:
            # anchor =  [0.333, 0.333, 0.334]
            anchor_pos = anchor[0] * vert_vector[0] + anchor[1] * vert_vector[1] + anchor[2] * vert_vector[2]
            for item in camera_items:
                ray_origin, ray_begin_local, bvhtree = item
                ray_direction = anchor_pos - ray_origin
                ray_direction.normalize()
                position, norm, faceID, _ = bvhtree.ray_cast(ray_begin_local, ray_direction, 50)
                if idx == faceID :
                    if norm != Vector ((0.0000, 0.0000, 0.0000)) and sum(norm * ray_direction) < 0:
                        observed = True
                        # print( "norm , ray_direction", norm , ray_direction,  )
                        # print ( " faceID, norm, ray_direction, norm.angle(ray_direction)>1.57 ",  faceID, norm, ray_direction, norm.angle(ray_direction)>1.57 )
                        break

            if observed:
                break

        if not observed :
            faces_select.append(face)


    bmesh.ops.delete(bm, geom=faces_select, context="FACES")
    bmesh.update_edit_mesh(me)
    bm.free()
    bpy.ops.object.mode_set(mode='OBJECT')

    return themesh





def run () :




    def compute_target_num_triangle ( size ):
        size = 2.6 if size > 2.6 else size
        size = 0.5 if size < 0.5 else size
        target = 1764 * size + 470
        return target


    # remove all default objects in a collection
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    import sys

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    print(argv)
    ply_path = argv [0]



    # bpy.ops.wm.ply_import(filepath="/path/to.ply")

    bpy.ops.import_mesh.ply(filepath=ply_path )

    the_mesh = None
    for ind, obj in enumerate(bpy.data.objects):
        if obj.type == 'MESH':
            the_mesh = obj


    obj_center, length = compute_mesh_size([the_mesh], skip=3)
    print("obj_center, length", obj_center, length)


    target_faces_num = compute_target_num_triangle( length )


    s_time = time.time()
    cameras = put_cam_around_obj(n_cam=100, obj_center=obj_center, length=length, fix_kpts=None)


    # return
    anchors = [[0.33, 0.33, 0.34],
               [0.70, 0.15, 0.15],
               [0.15, 0.70, 0.15],
               [0.15, 0.15, 0.70],
               [0.98, 0.01, 0.01],
               [0.01, 0.98, 0.01],
               [0.01, 0.01, 0.98]]

    print( "# verts before,", len( the_mesh.data.vertices) )

    the_mesh = delete_occluded_faces_from_cameras(the_mesh, cameras, anchors= anchors)

    print( "# verts after,", len( the_mesh.data.vertices) )

    print("time:delete_occluded_faces_from_cameras:", time.time() - s_time)



    obj_name = ply_path.split("/")[-1][:-4]

    obj_path = Path ( ply_path )

    dump_patth = os.path.join( obj_path.parent  , obj_name  + "_rm_occlusion.ply")
    dump_patt_obj = os.path.join( obj_path.parent  , obj_name  + "_rm_occlusion.obj")

    # dump_patth.mkdir(parents=True, exist_ok=True)

    print( "dump_patth", dump_patth)

    # manifold_path = dump_patth /  (str(obj_name) + ".obj")
    # export_mesh_obj(the_mesh, str(dump_patth), 'AUTO')

    bpy.ops.export_scene.obj(filepath=dump_patt_obj)
    # bpy.ops.export_mesh.ply( filepath=dump_patth )





if __name__ == '__main__':



    run()

