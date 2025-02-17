import torch
import os
import scipy
import scipy.sparse as sparse
import json
import open3d as o3d
import numpy as np
import trimesh
from pathlib import Path

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data



# indexs = str(Path(os.path.join(os.path.dirname(code_file_path))).parent / "bodyfit" / "smplx_vert_segmentation.json")


indexs = os.path.join( os.path.dirname( __file__ ), "smplx_vert_segmentation.json")





body_info = {}
with open(indexs) as f:
    indexs = json.load(f)
    head_index = indexs["head"]

    ear_index = indexs["right_ear"] + indexs["left_ear"]

    hair_index = [x for x in head_index if x not in ear_index]

    arms = ["leftArm", "rightArm",
            "leftForeArm", "rightForeArm",
            "leftHand", "rightHand",
            "leftHandIndex1", "rightHandIndex1",
            "rightShoulder", "leftShoulder"
            ]
    arms_index = []
    for ele in arms:
        arms_index = arms_index + indexs[ele]

    torsol = ["leftLeg", "leftToeBase", "leftFoot", "spine1", "spine2", "rightFoot", "rightLeg", "rightToeBase",
              "spine", "leftUpLeg", "hips", "rightUpLeg", "neck"]
    torsol_index = []
    for ele in torsol:
        torsol_index = torsol_index + indexs[ele]

    left_shoe = ["leftLeg", "leftToeBase", "leftFoot", "leftUpLeg"]
    left_shoe_index = []
    for ele in left_shoe:
        left_shoe_index = left_shoe_index + indexs[ele]
    left_foot_index = indexs["leftFoot"]

    right_shoe = ["rightLeg", "rightToeBase", "rightFoot", "rightUpLeg"]
    right_shoe_index = []
    for ele in right_shoe:
        right_shoe_index = right_shoe_index + indexs[ele]
    right_foot_index = indexs["rightFoot"]


    full_cloth_index = torsol_index + arms_index

    part_index_map = {
        "hair": hair_index,
        "left_shoe": left_shoe_index,
        "right_shoe": right_shoe_index,
        "left_foot": left_foot_index,
        "right_foot": right_foot_index,
        "torsol": torsol_index,
        "full_cloth": torsol_index + arms_index
    }

    part_wrap_ref_index_map = {
        "top": full_cloth_index,
        "outfit": full_cloth_index,
        "trousers": full_cloth_index,
        "others": full_cloth_index,
        "hair": hair_index,
        "l-shoe": left_shoe_index,
        "r-shoe": right_shoe_index,

    }


    part_scale_ref_index_map = {
        "top": torsol_index,
        "outfit": torsol_index,
        "trousers": torsol_index,
        "others": torsol_index,
        "hair": hair_index,
        "l-shoe": left_shoe_index,
        "r-shoe": right_shoe_index,
    }
