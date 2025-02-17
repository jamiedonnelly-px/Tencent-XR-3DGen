import glob
import multiprocessing
import os, glob
import sys
from pathlib import Path
import string
import multiprocessing
import numpy as np
import json

# blender = '/apdcephfs_data_cq3/share_2909871/rabbityli/soft/blender-2.92.0-linux64/blender'

blender = "/root/blender-3.5.0-linux-x64/blender"

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data


base_body_map = load_json( "./body_config/T_bodies.json")
base_body_map_source = load_json("./body_config/base_body_map_source.json")
key_smpl_map = load_json( "/aigc_cfs_2/rabbityli/bodyfit/webui/20240711_key_smpl_map.json" )

dump_root = "./output"





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
            else:
                file_found = False
                return False

        return file_found

    else :
        return False



def process_one_obj( json_path, obj_json, msg):

    print("------------------------------------------------------------------------------", msg, "")
    json_object = json.dumps(obj_json, indent=4)
    with open(json_path, "w") as f:
        f.write(json_object)

    cmd = " ".join([
        "python",
        "cloth_warpper.py",
        "--lst_path", json_path
    ])

    print(cmd)

    os.system(cmd)



def ruku_2_map ( ruku_ok ):
    ruku_ok = load_json(ruku_ok)["data"]
    clses = ruku_ok.keys()
    ruku_map = {}
    for cls in clses:
        ruku_map.update( ruku_ok[cls] )
    return  ruku_map


if __name__ == '__main__':

    maps = []

    shoe_map = ruku_2_map( "./part/shoe.json")
    shoe_keys = list( shoe_map.keys() )

    outfit_map = ruku_2_map( "./part/outfit.json")
    outfit_keys = list( outfit_map.keys() )

    trousers_map = ruku_2_map( "./part/trousers.json")
    trousers_keys = list( trousers_map.keys())

    top_map = ruku_2_map( "./part/top.json")
    top_keys = list( top_map.keys() )

    hair_map = ruku_2_map( "./part/hair.json")
    hair_keys = list(hair_map.keys())



    # for i in [ 2 ]:
    # for i in range(500):
    for i in range(len(hair_keys)):

        # if i % 7 != 0 :
        #     continue

        # for bk in ["pubg", "timer", "quest_strong", "yuanmeng"]:

        obj_json = {
            "path": {},
            "body_attr": ["male", "pubg"]
        }

        # obj_json["path"] ["trousers_keys"] = { "asset_key": trousers_keys[i] }
        # obj_json["path"] ["outfit_keys"] = { "asset_key": outfit_keys[i] }
        # obj_json["path"] ["shoe_keys"] = { "asset_key": shoe_keys[i] }
        # obj_json["path"] ["top_keys"] = { "asset_key": top_keys[i] }
        obj_json["path"] ["hair_keys"] = { "asset_key": hair_keys[i] }

        # name = "-".join( [trousers_keys[i], outfit_keys[i], shoe_keys[i]] )
        # name = "-".join( [trousers_keys[i], outfit_keys[i]] )

        # json_path = os.path.join("./jsons", str(i) + "---" + name + ".json" )
        json_path = os.path.join("./jsons",  hair_keys[i] + ".json" )

        maps.append([json_path, obj_json])



    # print( len(maps))
    # maps = maps[:10]
    # print(maps)
    # exit(0)


    sync = True
    if sync:
        pool = multiprocessing.Pool(12, maxtasksperchild=4)


    # maps = maps[10:]


    for idx,  ele in enumerate(maps):


        msg = str(idx) + "/" + str(len(maps))

        json_path, obj_json = maps[idx]

        if sync:

            res = pool.apply_async(process_one_obj, args=(json_path, obj_json, msg))

        else:

            process_one_obj(json_path, obj_json, msg)

            break

    if sync:
        pool.close()
        pool.join()