import copy
import importlib.util
import os, sys
import argparse
import shutil
from glob import glob
import trimesh
import torch
import pathlib
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import cv2
import scipy
import trimesh
import time

import numpy as np
from scipy import sparse

# from lib.timer import Timers
# timers = Timers()

import json
from utils.utils import load_json

import open3d as o3d
import numpy as np
from pathlib import Path

from body import Body
from asset import Asset
from transfer import transfer_cloth, save_cloth


# code_file_path = "/aigc_cfs_2/rabbityli/bodyfit/webui/cloth_warpper.py"
# base_body_map = load_json(os.path.join(os.path.dirname(code_file_path), "base_body_map.json"))
# base_body_map_source = load_json(os.path.join(os.path.dirname(code_file_path), "base_body_map_source.json"))

base_body_map = load_json( "./body_config/base_body_map.json")
base_body_map_source = load_json("./body_config/base_body_map_source.json")


def main():

    global body_info


    key_smpl_map = load_json( "/aigc_cfs_2/rabbityli/bodyfit/webui/20240711_key_smpl_map.json" )


    parser = argparse.ArgumentParser()
    parser.add_argument("--lst_path", type=str, required=True)
    args = parser.parse_args()

    lst_path = args.lst_path

    dump_root = Path(lst_path).parent

    print("---------------inside warp_clothes()---------------")
    print(lst_path)



    with open(lst_path, "rb") as f:
        data = json.load(f)
        part_info = data["path"]
        body_info = data["body_attr"]

        print("body_info", body_info)

    if len(part_info) == 0:
        print("no items in the list")
        exit()


    device = torch.device(0)

    # timers.tic('load smplx')
    # smplxs = SMPL_with_scale(cfg).to(torch.device(0))
    # timers.toc('load smplx')

    G_trns = np.eye(4)
    G_trns[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()

    print( "body_info", body_info )

    # body_info[0] = "male"
    # body_info[1] = "yuanmeng"
    # print(base_body_map)
    target_dir = base_body_map [body_info[0]][body_info[1]]["path"]
    # target_dir = "/aigc_cfs/rabbityli/base_bodies/MCWY2_F_T"

    target_body = os.path.join(target_dir, "smplx_and_offset_smplified.npz")
    target_body_manifold = os.path.join(target_dir, "warpped_smpl.obj")
    print("target_body",target_body)

    T_body = Body(target_body,  device=device, body_manifold= target_body_manifold)

    warp_lst = {}

    for idx, part in enumerate(part_info):


        label = part_info[part]["cat"]  # categories


        #check hair and shoes:
        if label == "shoe" and base_body_map[body_info[0]][body_info[1]]["use_shoes"]==False:
            print("use_shoes False , skip")
            continue
        if label == "hair" and base_body_map[body_info[0]][body_info[1]]["use_hair"] == False:
            print("use_hair False , skip")
            continue



        # if label == "others"

        name = "part_" + "%02d" % idx
        dump_dir = os.path.join(dump_root, name)
        Path(dump_dir).mkdir(exist_ok=True, parents=True)




        if label == "shoe":
            dump_path = os.path.join(dump_dir, name )
        else :
            dump_path = os.path.join(dump_dir, name + ".obj")

        warp_lst[dump_dir] = label

        print("part", part)

        body_key = part_info[part]["key"]  # body key
        if body_key[1] in ["daz", "vroid"]:
            # asset_body = os.path.join( pathlib.Path (part).parent.parent, "smplx_and_offset_smplified.npz" )
            asset_key = part_info[part]["asset_key"]
            asset_body = key_smpl_map[ asset_key]

        else:
            nake_dir = base_body_map_source[body_key[0]][body_key[1]]["path"]
            asset_body = os.path.join(nake_dir, "smplx_and_offset_smplified.npz")

        print("asset_body" , asset_body)

        proxy_path = part_info[part]["proxy_path"]


        asset = Asset(proxy_path, part, G_trns, label="cloth", device = device)
        S_body = Body(asset_body, device=device)
        m = transfer_cloth(S_body, T_body, asset)
        save_cloth( m , dump_path )

        # warp_one_cloth(part, proxy_path, asset_body, T, G_trns, dump_path, label)

    json_object = json.dumps(warp_lst, indent=4)
    with open(os.path.join(dump_root, "warp_lst.json"), "w") as f:
        f.write(json_object)

    with open(os.path.join(dump_root, "smplx-path.txt"), "w") as f:
        f.write(f"{target_dir}\n")


if __name__ == '__main__':
    # timers.tic('total')
    main()
    # timers.toc('total')
    # timers.print()