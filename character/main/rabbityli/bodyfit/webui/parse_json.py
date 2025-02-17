import os
import glob
from pathlib import Path
import json

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

def pprint(j):
    j = json.dumps(j, indent=2)
    print( j )

old = "/aigc_cfs_2/xiaqiangdai/project/objaverse_retrieve/mcwy_data_withObj_20240202.json"
old = load_json(old)

new = "/aigc_cfs_2/xiaqiangdai/project/objaverse_retrieve/layer_embedding_20240327_mcwy1.json_category.json"
new_copy = load_json( new )
new = load_json(new)


key_map = "/aigc_cfs/Asset/lists/layer/key_match.json"
key_map = load_json(key_map)

# import pdb
# pdb.set_trace()
# pprint( key_map )


cnt_t = 0
cnt_f = 0
# loop over new keys
for cls in new["data"].keys():
    for brand in new["data"][cls].keys():
        for new_key in new["data"][cls][brand].keys():


            if new_key in key_map.keys():
                mapped_old_key = key_map[new_key]

                nake_exist=False
                key_exist=False

                # loop over old keys
                for brand_old in old["data"].keys():
                    tag = False
                    for old_key in old["data"][brand_old].keys():


                        if mapped_old_key == old_key:
                            part = old["data"][brand_old] [ mapped_old_key ]["Obj_Mesh"]
                            nake =  str(Path(part).parent / "nake_model" )
                            body_key = str(Path(part).parent / "nake_model" / "body_key.txt")

                            if  os.path.exists( nake ) :
                                # print( "nake", nake)
                                nake_exist = True

                                cnt_t += 1

                                with open(body_key, "rb") as file:
                                    body_key = file.read().decode().rstrip().split(" ")
                                    # print( "body_key:", body_key )
                                    new_copy["data"][cls][brand][new_key]["body_key"] = body_key

                            else :
                                new_copy["data"][cls][brand].pop(new_key)
                                cnt_f += 1



                            print( "cnt_t, cnt_f", cnt_t, cnt_f)
                            tag = True
                            key_exist = True
                            break



                    if tag:
                        break

                if not key_exist:
                    new_copy["data"][cls][brand].pop(new_key)
                    # new_copy["data"][cls][brand][new_key] = ['', '']



new_copy_path = "/aigc_cfs_2/xiaqiangdai/project/objaverse_retrieve/layer_embedding_20240327_mcwy1.json_category.body_key.json"
json_object = json.dumps(new_copy, indent=4)
with open( new_copy_path, "w") as outfile:
    outfile.write(json_object)
