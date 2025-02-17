import bpy
import bmesh
import  numpy as np
import os
from pathlib import Path
import string
import json

valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)

import sys
sys.path.insert(0, "..")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def join_list_of_mesh ( mesh_list ):
    assert len(mesh_list) > 0
    if len(mesh_list) > 1 :
        bpy.ops.object.select_all(action='DESELECT')
        for ind, obj in enumerate(mesh_list):
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
        joint_mesh =  bpy.context.object
    else:
        joint_mesh = mesh_list[0]
    return joint_mesh

def solidify (the_mesh, thickness = 0.005):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    the_mesh.modifiers.new("solidify", "SOLIDIFY")
    the_mesh.modifiers["solidify"].thickness =thickness
    bpy.ops.object.convert(target='MESH') # bake modifier to mesh
    return  the_mesh


def decimate(the_mesh, target_faces_num) :
    num_faces = len(the_mesh.data.polygons)
    if num_faces > target_faces_num:
        decimate_ratio = target_faces_num / num_faces
        decimator = the_mesh.modifiers.new("decimate", "DECIMATE")
        decimator.ratio = decimate_ratio
        bpy.ops.object.convert(target='MESH')
    return the_mesh


def remesh (the_mesh, voxel_size):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    the_mesh.modifiers.new("remesh", "REMESH")
    the_mesh.modifiers["remesh"].voxel_size = voxel_size
    the_mesh.modifiers.new("triangulate", "TRIANGULATE")
    print( "the_mesh.mode", the_mesh.mode)
    # if the_mesh.mode == 'Edit':
    #     bpy.ops.object.editmode_toggle()
    print( "the_mesh.mode", the_mesh.mode)
    bpy.ops.object.convert(target='MESH') # bake modifier to mesh
    return the_mesh



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

def export_mesh_obj( mesh, mesh_path, path_mode= 'STRIP', global_scale=1 ):
    print( "export mesh", mesh, "# triangles", len(mesh.data.polygons))
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.export_scene.obj(filepath=mesh_path,
                             use_selection=True,
                             path_mode=path_mode,
                             global_scale=global_scale)
    bpy.ops.object.select_all(action='DESELECT')
    return mesh




def run (obj_path, dump_root) :


    # remove all default objects in a collection
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    obj_name = obj_path.split("/")[-1][:-4]


    thickness_standard = 0.02
    voxel_size_standard = 0.013

    bpy.ops.import_scene.obj(filepath=obj_path)

    meshes = []
    for ind, obj in enumerate(bpy.data.objects):
        if obj.type == 'MESH':
            meshes.append(obj)
    the_mesh = join_list_of_mesh( meshes )


    print("the_mesh",the_mesh)



    obj_center, length = compute_mesh_size([the_mesh], skip=1 )
    print( "obj_center, length", obj_center, length )



    thickness = length * thickness_standard / 1.618
    voxel_size = length * voxel_size_standard / 1.618



    print( "thickness, voxel_size", thickness, voxel_size)
    print("obj_center, length", obj_center, length)

    print( "solidify")
    the_mesh = solidify( the_mesh,thickness=thickness)

    print( "remesh, convert to manifold mesh")
    the_mesh = remesh(the_mesh, voxel_size=voxel_size)


    # separate mesh via isolation check
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = the_mesh
    the_mesh.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')
    meshes = []
    for ind, obj in enumerate(bpy.data.objects):
        if obj.type == 'MESH':
            if len(obj.data.polygons) > 500:
                meshes.append(obj)


    for i in range ( len(meshes) ):

        mesh = meshes[i]

        manifold_root = Path(dump_root) / "manifold"
        manifold_root.mkdir(exist_ok=True, parents=True)
        manifold_path = os.path.join(  manifold_root,  "manifold-" + str(i) + ".obj" )
        export_mesh_obj(mesh, str(manifold_path), 'AUTO')



    manifold_info = {
        "thinkness": thickness,
        "parts": len(meshes)
    }

    print("manifold_info", manifold_info)

    blender_info = Path (dump_root) / "info.json"
    json_object = json.dumps(manifold_info, indent=4, cls=NumpyEncoder)
    with open( blender_info, "w") as outfile:
        outfile.write(json_object)








if __name__ == '__main__':

    import sys

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    print(argv)
    obj_path = argv [0]
    dump_root = argv [1]




    # obj_path = "/home/rabbityl/tboard/DR_394_F_A/DR_394_fbx2020.obj"
    # # obj_path = "/home/rabbityl/tboard/DR_394_F_A/"
    # dump_root = "/home/rabbityl/tboard/DR_394_F_A/proxy_mesh"

    run( obj_path, dump_root )

    #
    # import multiprocessing
    # import time
    #
    # start_time = time.time() #second
    #
    # p = multiprocessing.Process(target=run, args=(obj_path, dump_root), name="Foo")
    # p.start()
    #
    # while True:
    #
    #     time.sleep(1)
    #
    #     if time.time() - start_time > 5 * 60 :  # tolerant for 5 minutes
    #         print( "timeout triggered")
    #         if p.is_alive():
    #             p.terminate()
    #             p.join()
    #         tof = Path (os.path.join(dump_root, obj_name,  "timeout.txt") )
    #         tof.parent.mkdir(parents=True, exist_ok=True)
    #         with tof.open("w", encoding="utf-8") as f:
    #             f.write("timeout")
    #         break
    #
    #
    #     if not p.is_alive() :
    #         break


