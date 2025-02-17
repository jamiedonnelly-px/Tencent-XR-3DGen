from torch.utils.data import Dataset
import json
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
from torchvision.utils import make_grid 
import json
import random
import h5py
import numpy as np

h5_path = "/apdcephfs_cq8/share_1615605/Asset/objaverse/render/3d_diffusion/new_sample_20240829/part1/proc_data/pod_0/objaverse/7e80ff98af4c4cb4ab726f9eee6aab99/proc_data/geometry/sample.h5"

with h5py.File(h5_path, 'r') as h5file:
    rand_idx = np.random.randint(0, 450000)
    surface_points = np.asarray(h5file["surface_points"][rand_idx:rand_idx+50000])
    surface_normals = np.asarray(h5file["surface_normals"][rand_idx:rand_idx+50000])
print(surface_points.shape)
print(surface_points)
print(surface_normals)