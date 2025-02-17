import glob
import multiprocessing
import os, glob
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

def process_one_obj( mesh_path, dump_path, background=True):

    background = "--background" if background else " "


    Path(dump_path).mkdir(exist_ok=True, parents=True)


    # # Check if file already exist
    # if check_manifold_exist(dump_path):
    #     print("file already exsit", dump_path)
    #     return


    cmd = " ".join( [
        blender,
        background,
        "-P",
        "blender_manifold.py",
        "--",
        mesh_path,
        dump_path
    ] )

    print(cmd)
    os.system(cmd)

    if check_manifold_exist(dump_path):
        print("file creation succeed")
        return


if __name__ == '__main__':


    maps = []

    parts = [
        "top",
        "trousers",
        "shoe",
        "outfit",
        "others",
        "hair"
    ]

    for part in parts:

        # part = "hair"

        ruku_ok = "../jsonconfig/part_daz/" + part +".json"
        ruku_ok = load_json(ruku_ok)

        dump_root = "/aigc_cfs_gdp/Asset/proxy_meshes/" + part


        clses = ruku_ok["data"].keys()
        for cls in clses:
            ids = ruku_ok["data"][cls].keys()
            for id in ids :
                mesh_path = ruku_ok["data"][cls][id]["Obj_Mesh"]
                # print( mesh_path )
                dump_path = os.path.join( dump_root,  id)
                maps.append( [ mesh_path, dump_path] )



    # print( len(maps))
    # exit(0)

    sync = True
    if sync:
        pool = multiprocessing.Pool(20, maxtasksperchild=4)

    for ele in maps:
        mesh_path, dump_path = ele
        if sync:
            res = pool.apply_async(process_one_obj, args=(mesh_path, dump_path, True))
        else:
            process_one_obj( mesh_path, dump_path, background=True)
            break

    if sync:
        pool.close()
        pool.join()