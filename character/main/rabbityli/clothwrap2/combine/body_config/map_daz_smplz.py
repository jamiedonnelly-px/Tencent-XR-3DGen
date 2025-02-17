import os, sys, glob
import json
import  pathlib

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

def write_json(fname , data  ):
    json_object = json.dumps(data, indent=4)
    with open(fname, "w") as outfile:
        outfile.write(json_object)

clses = [ "DAZ_DAZ_Top", "DAZ_DAZ_Outfit", "DAZ_DAZ_Bottom"]



query_json = "../part/20240925_gdp.json"
query_json = load_json(query_json)

data_json = "/aigc_cfs_11/Asset/active_list/layered_data/20240618/layer_embedding_20240618_all_with_all_glb.json"
data_json = load_json(data_json)

root = "/aigc_cfs_gdp/Asset/daz_bodies"

cnt = 0
non = 0


key_smpl_map = {}

for cls in clses:
    for key in query_json["data"][cls].keys():

        obj_path = data_json["data"][cls][key]["Mesh"]



        smpl_path = os.path.join( pathlib.Path(obj_path).parent.parent.parent, "split/", "smplx_and_offset_smplified.npz")

        smpl_path = smpl_path.replace("(", "\(")
        smpl_path = smpl_path.replace("&", "\&")
        smpl_path = smpl_path.replace(")", "\)")

        dump_dir= os.path.join( root, key)
        pathlib.Path ( dump_dir ).mkdir( parents=True, exist_ok=True)

        dump_path = os.path.join(dump_dir, "smplx_and_offset_smplified.npz")


        key_smpl_map[key] = dump_path




        if os.path.exists(dump_path):
            cnt += 1
        else :
            cmd = " ".join(
                [
                    "cp",
                    smpl_path,
                    dump_path
                ]
            )

            print( cmd )

            os.system( cmd )


        print( "--------------------------------")
        print( cnt)
        print( non)


write_json("./daz-map.json", key_smpl_map)