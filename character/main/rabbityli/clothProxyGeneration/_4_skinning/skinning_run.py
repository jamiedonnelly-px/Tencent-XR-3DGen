import glob
import multiprocessing
import os, glob
import  pathlib
import sys
from pathlib import Path
import string
import multiprocessing

import json

# blender = '/apdcephfs_data_cq3/share_2909871/rabbityli/soft/blender-2.92.0-linux64/blender'

blender = "/root/blender-3.5.0-linux-x64/blender"


VERT_LIMIT = {
    # "top": 4000,
    "trousers": 3500,
    # "shoe": 1500,
    # "outfit": 5000,
    # "others": 4000,
    # "hair": 3000
}


def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data


def check_proxy_exist(proxy_flder):
    dump_info = os.path.join(proxy_flder, "info.json")
    if os.path.exists(dump_info):
        npart= load_json(dump_info)["parts"]
        file_found = False
        for i in range (npart) :
            manifold = os.path.join( proxy_flder , "proxy", "part-"+str(i)+".ply")
            laplacian = os.path.join( proxy_flder , "proxy", "part-"+str(i)+"-laplacian.npz")
            if os.path.exists( manifold) and os.path.exists( laplacian):
                file_found = True
            else :
                file_found = False
                return False

        return file_found

    else :
        return False

def process_one_obj(  visual_path, proxy_flder , msg ):

    print("--------------------------", msg , "--------------------------")



    cmd = " ".join([
        "python",
        "skinning.py",
        "--v", visual_path,
        "--p", proxy_flder])


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
            for id in ids:

                # if  ruku_ok["data"][cls][id]["replace_type"] == "replace" :


                if ruku_ok["data"][cls][id]["Category"] == "shoe" :


                   # pass


                    visual_path = Path(ruku_ok["data"][cls][id]["Obj_Mesh"]).parent
                    proxy_flder = os.path.join(proxy_root, id)


                    # left
                    left_proxy_flder = os.path.join(proxy_flder, "left")
                    left_mesh = os.path.join(visual_path, "left/asset.obj")

                    # right
                    right_proxy_flder = os.path.join(proxy_flder, "right")
                    right_mesh = os.path.join(visual_path, "right/asset.obj")

                    maps.append([left_mesh, left_proxy_flder])
                    maps.append([right_mesh, right_proxy_flder])



                else :
                    # pass
                    print( id )
                    visual_path = ruku_ok["data"][cls][id]["Obj_Mesh"]
                    proxy_flder = os.path.join( proxy_root, id )


                    maps.append( [visual_path, proxy_flder] )

    print( len(maps ))

    # exit()

    sync = True
    if sync:
        pool = multiprocessing.Pool(60, maxtasksperchild=4)


    for idx,  ele in enumerate(maps):

        msg = str(idx) + "/" + str(len(maps))

        visual_path, proxy_flder = ele

        print("proxy_flder", proxy_flder)
        if sync:
            res = pool.apply_async(process_one_obj, args=( visual_path, proxy_flder, msg))
        else:
            process_one_obj(visual_path, proxy_flder, msg)


            if idx >2: break

    if sync:
        pool.close()
        pool.join()