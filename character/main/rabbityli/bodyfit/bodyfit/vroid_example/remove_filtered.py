import pdb

import trimesh
import os
import glob
import numpy as np
import pathlib
import json


def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data


def main(data_json):

    data_json = load_json(data_json)

    clses = ["VRoid_VRoid_Top", "VRoid_VRoid_Shoe", "VRoid_VRoid_Hair", "VRoid_VRoid_Top"]

    screenshot_root = "/aigc_cfs_2/rabbityli/screenshot_vroid"

    trash_box = "/aigc_cfs_2/rabbityli/screenshot_trashbox"
    pathlib.Path(trash_box).mkdir( exist_ok=True)


    images = glob.glob( os.path.join(screenshot_root, "*", "*jpg") )

    cnt = 0
    for img_path in images:
        img = img_path.split("/")
        asset = img[-1][:-4]
        cls = img[-2]

        cnt += 1
        print( cnt, "/", len(images))
        if asset in data_json["data"][cls] :
            # print( "in", asset )
            pass
        else :
            # print( "out", asset)

            cmd = " ".join(
                [
                    "mv",
                    img_path,
                    trash_box
                ]
            )

            print ( cmd )

            os.system(cmd)



if __name__ == '__main__':


    data_json = "../../json_manager/jsons/sampled_vroid.json"

    main(data_json)