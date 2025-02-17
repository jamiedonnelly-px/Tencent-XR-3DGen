import glob
import multiprocessing
import os, glob
import pathlib
import sys
from pathlib import Path
import string
import multiprocessing

import json

# blender = '/apdcephfs_data_cq3/share_2909871/rabbityli/soft/blender-2.92.0-linux64/blender'

blender = "/root/blender-3.5.0-linux-x64/blender"




def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data


def check_manifold_exist(dump_path):
    dump_info = os.path.join(dump_path, "info.json")
    if os.path.exists(dump_info):
        npart= load_json(dump_info)["parts"]
        file_found = False
        for i in range (npart) :
            manifold = os.path.join( dump_path , "manifold", "manifold-"+str(i)+".obj")
            if os.path.exists( manifold):
                file_found = True
            else :
                file_found = False
                return False

        return file_found

    else :
        return False

def process_one_obj( mesh_path, dump_path, msg, background=True):

    print("--------------------------", msg , "--------------------------")


    background = "--background" if background else " "


    Path(dump_path).mkdir(exist_ok=True, parents=True)


    # Check if file already exist
    # if check_manifold_exist(dump_path):
    #     print("file already exsit", dump_path)
    #     return


    cmd = " ".join( [
        blender,
        background,
        "-P",
        "../_1_manifold/blender_manifold.py",
        "--",
        mesh_path,
        dump_path
    ] )

    # print(cmd)
    os.system(cmd)
    #
    # if check_manifold_exist(dump_path):
    #     print("file creation succeed")
    #     return


if __name__ == '__main__':

    part = "shoe"

    ruku_ok = "../jsonconfig/part/" + part +".json"
    ruku_ok = load_json(ruku_ok)

    dump_root = "/aigc_cfs_gdp/Asset/proxy_meshes/" + part

    maps = []

    clses = ruku_ok["data"].keys()
    for cls in clses:
        ids = ruku_ok["data"][cls].keys()
        for id in ids :
            mesh_path = pathlib.Path( ruku_ok["data"][cls][id]["Obj_Mesh"]).parent
            dump_path = os.path.join( dump_root,  id)

            # left
            left_mesh = os.path.join(mesh_path, "left/asset.obj")
            lft_dump_path = os.path.join( dump_root,  id, "left")

            # right
            right_mesh = os.path.join(mesh_path, "right/asset.obj")
            right_dump_path = os.path.join( dump_root,  id, "right")

            maps.append( [ left_mesh, lft_dump_path] )
            maps.append( [ right_mesh, right_dump_path] )



    sync = True
    if sync:
        pool = multiprocessing.Pool(30, maxtasksperchild=4)

    for idx, ele in enumerate( maps):
        mesh_path, dump_path = ele
        msg = str(idx) + "/" + str(len(maps))

        if sync:
            res = pool.apply_async(process_one_obj, args=(mesh_path, dump_path,msg, True))
        else:
            process_one_obj( mesh_path, dump_path, msg, background=True)


    if sync:
        pool.close()
        pool.join()