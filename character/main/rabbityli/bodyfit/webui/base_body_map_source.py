from base_body_map  import base_body_map
import json

base_body_map_source = base_body_map.copy()



base_body_map_source ["male"]["readyplayerme"] = {
    "path": "/aigc_cfs/rabbityli/base_bodies/readyplayerme_male",
    "shoes_scale": 1.0
}

base_body_map_source ["male"]["readyplayerme_T"] = {
    "path": "/aigc_cfs/rabbityli/base_bodies/readyplayerme_male_T",
    "shoes_scale": 1.0
}


base_body_map_source ["male"]["mcwy2_A"] = {
    "path": "/aigc_cfs/rabbityli/base_bodies/MCWY2_M_A",
    "shoes_scale": 1.0
}

base_body_map_source ["female"]["mcwy2_A"] = {
    "path": "/aigc_cfs/rabbityli/base_bodies/MCWY2_F_A",
    "shoes_scale": 1.0
}


base_body_map_source ["female"]["mcwy1_old"] = {
    "path": "/aigc_cfs/rabbityli/base_bodies/mcwy1_M_old",
    "shoes_scale": 1.0
}
base_body_map_source ["male"]["mcwy1_old"] = {
    "path": "/aigc_cfs/rabbityli/base_bodies/mcwy1_M_old",
    "shoes_scale": 1.0
}



def load_json(j):
    with open(j) as f:
        data = json.load(f)
    return data

def write_json(fname,j):
    json_object = json.dumps(j, indent=4)
    with open( fname, "w") as outfile:
        outfile.write(json_object)



if __name__ == '__main__':



    write_json( "./base_body_map_source.json"  ,base_body_map_source )

