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


base_body_map = load_json( "../_5_SDF/body_config/base_body_map.json")
base_body_map_source = load_json("../_5_SDF/body_config/base_body_map_source.json")
key_smpl_map = load_json( "/aigc_cfs_2/rabbityli/bodyfit/webui/20240711_key_smpl_map.json" )


VERT_LIMIT = {
    # "top": 4000,
    # "trousers": 3500,
    "shoe": 1000,
    # "outfit": 5000,
    # "others": 4000,
    # "hair": 3000
}




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

def process_one_obj( proxy_flder, asset_body, msg):

    print("--------------------------", msg , "--------------------------")



    cmd = " ".join([
        "python",
        "../_5_SDF/sdf.py",
        "--b", asset_body,
        "--p", proxy_flder])


    # print(cmd)

    os.system(cmd)


# def obtain_body_path( ):



if __name__ == '__main__':

    maps = []

    for part in VERT_LIMIT.keys():

        # print ("part", part)

        ruku_ok = "../jsonconfig/part/" + part +".json"
        ruku_ok = load_json(ruku_ok)

        proxy_root = "/aigc_cfs_gdp/Asset/proxy_meshes/" + part




        i = 0

        clses = ruku_ok["data"].keys()
        for cls in clses:
            ids = ruku_ok["data"][cls].keys()
            for id in ids:

                body_key = ruku_ok["data"][cls][id]["body_key"]

                if body_key[1] in [ "vroid"]:
                    # asset_body = os.path.join( pathlib.Path (part).parent.parent, "smplx_and_offset_smplified.npz" )
                    asset_body =  key_smpl_map[id]

                else:
                    nake_dir = base_body_map_source[body_key[0]][body_key[1]]["path"]
                    asset_body = os.path.join(nake_dir, "smplx_and_offset_smplified.npz")




                proxy_flder = os.path.join( proxy_root, id)




                # left
                left_proxy_flder = os.path.join(proxy_flder, "left")

                # right
                right_proxy_flder = os.path.join(proxy_flder, "right")


                maps.append( [ left_proxy_flder, asset_body] )
                maps.append( [ right_proxy_flder, asset_body] )

    # print( len(maps ))


    sync = True
    if sync:
        pool = multiprocessing.Pool(15, maxtasksperchild=4)


    for idx,  ele in enumerate(maps):

        msg = str(idx) + "/" + str(len(maps))

        proxy_flder, asset_body = ele

        print("proxy_flder", proxy_flder)
        if sync:
            res = pool.apply_async(process_one_obj, args=( proxy_flder, asset_body, msg))
        else:
            process_one_obj(proxy_flder, asset_body, msg)

            break

            # if idx >2: break

    if sync:
        pool.close()
        pool.join()