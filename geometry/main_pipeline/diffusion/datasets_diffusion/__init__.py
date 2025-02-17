
from datasets_diffusion.datasets_diffusion3d_v6_1cond import Diffusion3D_V6_1Cond
from datasets_diffusion.datasets_diffusion3d_v6_4cond import Diffusion3D_V6_4Cond


def get_dataset(configs, data_type="train", resample=True, load_from_cache_last=False, **kwargs):
    if configs["data_config"]["dataset_name"] == "Diffusion3D_V6_1Cond":
        return Diffusion3D_V6_1Cond(configs, data_type=data_type, **kwargs)
    elif configs["data_config"]["dataset_name"] == "Diffusion3D_V6_4Cond":
        return Diffusion3D_V6_4Cond(configs, data_type=data_type, **kwargs)
