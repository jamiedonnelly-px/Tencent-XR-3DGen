import glob
import multiprocessing
import os, glob
import sys
from pathlib import Path
import string
import multiprocessing

import json

# blender = '/apdcephfs_data_cq3/share_2909871/rabbityli/soft/blender-2.92.0-linux64/blender'



VERT_LIMIT = {
    # "top": 4000,
    # "trousers": 3500,
    "shoe": 1500,
    # "outfit": 5000,
    # "others": 4000,
    # "hair": 3500
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

def process_one_obj( visual_path, msg ):

    print("--------------------------", msg , "--------------------------")


    cmd = " ".join( [
        "python",
        "split.py",
        "--v", visual_path
    ])
    print(cmd)
    os.system(cmd)




if __name__ == '__main__':


    ruku_ok = "../jsonconfig/part/shoe.json"
    ruku_ok = load_json(ruku_ok)


    i = 0

    maps = []

    clses = ruku_ok["data"].keys()
    for cls in clses:
        ids = ruku_ok["data"][cls].keys()
        for id in ids :



            # print( id )
            visual_path = ruku_ok["data"][cls][id]["Obj_Mesh"]


            maps.append( visual_path  )




    sync = True
    if sync:
        pool = multiprocessing.Pool(10, maxtasksperchild=4)

    for idx, ele in enumerate( maps ):
        visual_path = ele
        msg = str(idx) + "/" + str(len(maps))

        if sync:
            res = pool.apply_async(process_one_obj, args=(visual_path, msg))
        else:
            process_one_obj(visual_path,  msg)
            break

    if sync:
        pool.close()
        pool.join()