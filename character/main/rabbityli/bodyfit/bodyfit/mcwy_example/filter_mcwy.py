import os
import glob
import pathlib
from pathlib import Path
import json

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

def write_json(fname , data  ):
    json_object = json.dumps(data, indent=4)
    with open(fname, "w") as outfile:
        outfile.write(json_object)

def pprint(j):
    j = json.dumps(j, indent=2)
    print( j )






if __name__ == '__main__':

    all = "./jsons/web_flatten_gdp.json"
    all = load_json ( all )

    # print( all)

    # for k in all.keys():
    #     if all[k]["body_key"][1] == "mcwy2_A":
    #         print(k)
