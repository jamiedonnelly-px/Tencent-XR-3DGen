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


VERT_LIMIT = {
    "top": 4000,
    "trousers": 3500,
    "shoe": 1500,
    "outfit": 5000,
    "others": 4000,
    "hair": 4000
}


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
            manifold = os.path.join( dump_path , "single_layer_manifold", "manifold-"+str(i)+".obj")
            if os.path.exists( manifold):
                file_found = True
            else :
                file_found = False
                return False

        return file_found

    else :
        return False

def process_one_obj( visual_path, proxy_flder, vert_limit):
    cmd = " ".join( [
        "python",
        "single_layer.py",
        "--v", visual_path,
        "--p", proxy_flder,
        "--n", str(vert_limit)
    ] )
    print(cmd)
    os.system(cmd)




if __name__ == '__main__':

    maps = []

    for part in VERT_LIMIT.keys():

        # print ("part", part)

        ruku_ok = "../jsonconfig/part_daz/" + part +".json"
        ruku_ok = load_json(ruku_ok)

        proxy_root = "/aigc_cfs_gdp/Asset/proxy_meshes/" + part


        # visual_path = args.v
        # proxy_flder = args.p
        # vert_limit = args.n


        i = 0

        clses = ruku_ok["data"].keys()
        for cls in clses:
            ids = ruku_ok["data"][cls].keys()
            for id in ids :



                # print( id )
                visual_path = ruku_ok["data"][cls][id]["Obj_Mesh"]
                proxy_flder = os.path.join( proxy_root, id )
                vert_limit = VERT_LIMIT[part]


                maps.append( [ visual_path, proxy_flder, vert_limit ] )



    sync = True
    if sync:
        pool = multiprocessing.Pool(30, maxtasksperchild=4)

    for ele in maps:
        visual_path, proxy_flder, vert_limit = ele
        if sync:
            res = pool.apply_async(process_one_obj, args=(visual_path, proxy_flder, vert_limit))
        else:
            process_one_obj(visual_path, proxy_flder, vert_limit)
            break

    if sync:
        pool.close()
        pool.join()