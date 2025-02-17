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


VERT_LIMIT = {
    # "top": 4000,
    # "trousers": 3500,
    "shoe": 2024

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

def process_one_obj( visual_path, proxy_flder, vert_limit, msg):

    print("--------------------------", msg , "--------------------------")


    cmd = " ".join( [
        "python",
        "../_2_single_layer_manifold/single_layer.py",
        "--v", visual_path,
        "--p", proxy_flder,
        "--n", str(vert_limit)
    ] )
    print(cmd)
    os.system(cmd)




if __name__ == '__main__':

    maps = []

    for part in VERT_LIMIT.keys():

        ruku_ok = "../jsonconfig/part/" + part +".json"
        ruku_ok = load_json(ruku_ok)

        proxy_root = "/aigc_cfs_gdp/Asset/proxy_meshes/" + part


        i = 0

        clses = ruku_ok["data"].keys()
        for cls in clses:
            ids = ruku_ok["data"][cls].keys()
            for id in ids :

                # print( id )
                visual_path = pathlib.Path( ruku_ok["data"][cls][id]["Obj_Mesh"]).parent
                proxy_flder = os.path.join( proxy_root, id )
                vert_limit = VERT_LIMIT[part]

                # left
                left_proxy_flder = os.path.join(proxy_flder, "left")
                left_mesh = os.path.join(visual_path, "left/asset.obj")

                # right
                right_proxy_flder = os.path.join(proxy_flder, "right")
                right_mesh = os.path.join(visual_path, "right/asset.obj")


                maps.append( [ left_mesh, left_proxy_flder, vert_limit ] )
                maps.append( [ right_mesh, right_proxy_flder, vert_limit ] )




    sync = True
    if sync:
        pool = multiprocessing.Pool(30, maxtasksperchild=4)



    for idx, ele in enumerate(maps):
        msg = str(idx) + "/" + str(len(maps))

        visual_path, proxy_flder, vert_limit = ele


        if sync:
            res = pool.apply_async(process_one_obj, args=(visual_path, proxy_flder, vert_limit,msg))
        else:
            process_one_obj(visual_path, proxy_flder, vert_limit,msg)
            break

    if sync:
        pool.close()
        pool.join()