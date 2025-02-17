import pdb

import trimesh
import os
import glob
import numpy as np
import pathlib
import  json

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data


def main(data_json, rank):

    data_json = load_json(data_json)


    clses = [ "VRoid_VRoid_Top"  , "VRoid_VRoid_Shoe", "VRoid_VRoid_Hair",  "VRoid_VRoid_Top"]

    screenshot_root = "/aigc_cfs_2/rabbityli/screenshot_vroid"


    cnt = 0
    for cls in clses:

        screenshot_path = os.path.join(screenshot_root, cls)
        pathlib.Path(screenshot_path).mkdir(exist_ok=True, parents=True)

        idkeys = list( data_json["data"][cls].keys() )



        all = len(idkeys)

        print( "total_keys", all )


        for asset in idkeys   :

            
            cnt +=1
            print( asset, cnt, "/", all )


            obj = data_json["data"][cls][asset]["Mesh"]

            # check if body is processed
            objp = pathlib.Path(obj).parent.parent.parent
            body = glob.glob(os.path.join(objp, "split/body", "*body.obj") )[0]

            smpl_param_path = os.path.join(objp, "split/", "smplx_and_offset_smplified.npz")

            body = body.replace("(", "\(")
            body = body.replace("&", "\&")
            body = body.replace(")", "\)")

            smpl_param_path = smpl_param_path.replace("(", "\(")
            smpl_param_path = smpl_param_path.replace("&", "\&")
            smpl_param_path = smpl_param_path.replace(")", "\)")


            # registration
            if os.path.exists(smpl_param_path):
                print( "#################Body Exist################")
                clean_up = False
                if clean_up :
                    cmd = " ".join(
                        [
                            "rm ",
                            smpl_param_path
                        ]
                    )
                    print( cmd )

                    os.system(cmd)

            else :
                # print( "#################@@@@@@@@@@@@@@@##############")
                # print( asset, cnt, "/", all)
                python = "/root/env/auto_rig/bin/python"
                # python = "python"
                cmd = " ".join(
                    [
                        python,
                        "_1_align_vroid.py",
                        "--mesh_path",
                        body,
                        "--dump_path",
                        smpl_param_path,
                        "--rank", rank
                    ]
                )
                print( cmd )
                # print("save")
                os.system(cmd)


            # copy screen shot
            ssp = os.path.join( pathlib.Path(smpl_param_path).parent, "screenshot.jpg" )
            ssp_target = os.path.join( screenshot_path, asset + ".jpg" )
            cmd = " ".join(
                [
                    "cp",
                    ssp,
                    ssp_target
                ]
            )

            os.system( cmd )



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str )#, required=True)
    parser.add_argument("--rank", type=str) #, required=True)
    args = parser.parse_args()
    rank = args.rank
    data_json = args.data_json


    data_json = "../../json_manager/jsons/sampled_vroid.json"
    rank=0

    # data_json = "../../json_manager/jsons/20240704_ruku_not_ok.json"


    main(data_json, rank)