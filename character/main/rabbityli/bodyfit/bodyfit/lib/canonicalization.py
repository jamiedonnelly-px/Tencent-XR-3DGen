import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from .smpl_w_scale import SMPL_with_scale

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fit_scan.util import compute_truncated_chamfer_distance
from .lndp_deformer  import Deformer as LNDP_Deformer

import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import json
from easydict import EasyDict as edict
import pathlib

from .utils import util, lossfunc, rotation_converter
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


from .index_config import *


class AvatarCanonicalization():
    def __init__(self):
        super(AvatarCanonicalization, self).__init__()

        pass








