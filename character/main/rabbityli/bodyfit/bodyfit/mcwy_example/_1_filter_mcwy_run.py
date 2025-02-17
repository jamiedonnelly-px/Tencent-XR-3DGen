import copy
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

def write_json(fname , data  ):
    json_object = json.dumps(data, indent=4)
    with open(fname, "w") as outfile:
        outfile.write(json_object)

j_template = {
    "path": {
        "exmaple": {
            "cat": "hair",
            "key": [
                "female",
                "mcwy2_A"
            ]
        },
    },
    "body_attr": [
        "female",
        "mcwy2"
    ],
}

if __name__ == '__main__':




    screenshot_root = "/aigc_cfs_2/rabbityli/screenshot/mcwy_filer"

    all = "../../json_manager/jsons/web_flatten_gdp.json"
    all = load_json(all)

    python = "/root/env/auto_rig/bin/python"

    for k in all.keys():
        if all[k]["body_key"][1] == "mcwy2_A":
            if all[k]["Category"] in ["outfit", "top"] :


                wrap_path = os.path.join(screenshot_root, k )
                pathlib.Path( wrap_path ).mkdir(exist_ok=True, parents=True)


                mesh_path = all[k]["Obj_Mesh"]

                print(mesh_path)

                j_form = copy.deepcopy( j_template )

                j_form["path"][mesh_path] = j_form["path"].pop( "exmaple")
                j_form["path"][mesh_path]["cat"] = all[k]["Category"]

                # write json
                print(j_form)
                obj_lst = os.path.join( wrap_path, "obj_lst.txt")

                print(obj_lst)
                write_json( obj_lst , j_form)

                cmd = " ".join(
                    [
                        "python",
                        "/aigc_cfs_2/rabbityli/bodyfit/webui/cloth_warpper.py",
                        "--lst_path",
                        obj_lst
                    ]
                )

                os.system(cmd)

                # exit(0)
                #
                # print( cmd )

                # print( "total_keys", all )
                #
                #
                # for asset in idkeys   :
                #
                #
                #     cnt +=1
                #     print( asset, cnt, "/", all )
                #
                #
                #     obj = data_json["data"][cls][asset]["Mesh"]
                #
                #     # check if body is processed
                #     objp = pathlib.Path(obj).parent.parent.parent
                #     body = glob.glob(os.path.join(objp, "split/body", "*body.obj") )[0]
                #
                #     smpl_param_path = os.path.join(objp, "split/", "smplx_and_offset_smplified.npz")
                #
                #     body = body.replace("(", "\(")
                #     body = body.replace("&", "\&")
                #     body = body.replace(")", "\)")
                #
                #     smpl_param_path = smpl_param_path.replace("(", "\(")
                #     smpl_param_path = smpl_param_path.replace("&", "\&")
                #     smpl_param_path = smpl_param_path.replace(")", "\)")
                #
                #
                #     # registration
                #     if os.path.exists(smpl_param_path):
                #         print( "#################Body Exist################")
                #         clean_up = False
                #         if clean_up :
                #             cmd = " ".join(
                #                 [
                #                     "rm ",
                #                     smpl_param_path
                #                 ]
                #             )
                #             print( cmd )
                #
                #             os.system(cmd)
                #
                #     else :
                #         # print( "#################@@@@@@@@@@@@@@@##############")
                #         # print( asset, cnt, "/", all)
                #         python = "/root/env/auto_rig/bin/python"
                #         # python = "python"
                #         cmd = " ".join(
                #             [
                #                 python,
                #                 "_1_align_vroid.py",
                #                 "--mesh_path",
                #                 body,
                #                 "--dump_path",
                #                 smpl_param_path,
                #                 "--rank", rank
                #             ]
                #         )
                #         print( cmd )
                #         # print("save")
                #         os.system(cmd)
                #
                #
                #     # copy screen shot
                #     ssp = os.path.join( pathlib.Path(smpl_param_path).parent, "screenshot.jpg" )
                #     ssp_target = os.path.join( screenshot_path, asset + ".jpg" )
                #     cmd = " ".join(
                #         [
                #             "cp",
                #             ssp,
                #             ssp_target
                #         ]
                #     )
                #
                #     os.system( cmd )


