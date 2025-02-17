import pdb

import trimesh
import os
import glob
import numpy as np
import pathlib
import  json


json_list = glob.glob( "../../json_manager/jsons/job_jsons/*json" )

print( json_list)
import torch
n_device = torch.cuda.device_count()

ranklst = [0, 0, 1, 1, 2, 2, 3, 3]

for i in range (len(json_list)):

    rank = str( ranklst[i] )
    json = json_list[i]

    cmd = " ".join(
        [
            "python",
            "_1_align_vroid_run.py",
            "--rank",
            rank,
            "--data_json",
            json
        ]
    )

    print(cmd)