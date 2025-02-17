import os, sys

sys.path.append("/aigc_cfs/tinatchen/layer/auto_rig/layer")
from run import auto_rig_layer


example_path = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/generate/5645a1f2-54e9-504f-b412-28cc9ff2c55e"



cmd = " ".join(
    "python --lst_path",
    os.path.join(example_path, "object_lst.txt")
)


os.system(cmd)

auto_rig_layer ( example_path )
