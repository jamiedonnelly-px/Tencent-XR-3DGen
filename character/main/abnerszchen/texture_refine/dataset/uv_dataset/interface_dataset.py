
import os
import sys
from datasets import load_dataset

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

HF_HOME="/data5/sz/huggingface" # TODO
# HF_HOME="/aigc_cfs_3/layer_tex/huggingface" # TODO
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME

def load_uv_dataset(dataset_root, cache_dir=f"/data5/sz/huggingface/datasets"):
    """ref dataset/uv_dataset.uv_dataset.py

    Args:
        dataset_root: dataset root dir with train/val/test.json
        cache_dir: _description_. Defaults to f"/data5/sz/huggingface/datasets".

    Returns:
        _description_
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = os.path.dirname(cache_dir)
    
    dset_loader = os.path.join(project_root, "dataset/uv_dataset") 
    assert os.path.exists(dset_loader), dset_loader
    ds = load_dataset(str(dset_loader), 
                      data_dir=dataset_root, 
                      token=False, 
                      cache_dir=cache_dir)
    return ds