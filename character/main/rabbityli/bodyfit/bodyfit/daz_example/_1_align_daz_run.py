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


data_json = "/aigc_cfs_11/Asset/active_list/layered_data/20240618/layer_embedding_20240618_all_with_all_glb.json"
# data_json = "./layer_embedding_20240618_all_with_all_glb.json"
data_json = load_json(data_json)



clses = [ "DAZ_DAZ_Top", "DAZ_DAZ_Shoe", "DAZ_DAZ_Outfit", "DAZ_DAZ_Bottom"]

all = 1432
cnt = 0
for cls in clses:
    for asset in data_json["data"][cls].keys():
        cnt +=1
        # print( asset, cnt, "/", all)

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



        if os.path.exists(smpl_param_path):


            print( "#################Body Exist################")


            pass

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



            cmd = " ".join(
                [
                    "python",
                    "_1_align_daz.py",
                    "--mesh_path",
                    body,
                    "--dump_path",
                    smpl_param_path
                ]
            )

            print( cmd )


            # print("save")
            os.system(cmd)

