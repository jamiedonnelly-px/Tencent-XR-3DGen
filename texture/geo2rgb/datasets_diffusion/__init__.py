from datasets_diffusion.datasets_zero123plus_v3_2_090180270 import ObjaverseDatasetV3_2_090180270
from datasets_diffusion.datasets_zero123plus_v3_2_090180270_controlnet import ObjaverseDatasetV3_2_090180270_controlnet

def get_dataset(configs, data_type="train", resample=True, load_from_cache_last=False):
    if configs["data_config"]["dataset_name"] == "ObjaverseDatasetV3_2_090180270":
        return ObjaverseDatasetV3_2_090180270(configs, data_type=data_type)
    elif configs["data_config"]["dataset_name"] == "ObjaverseDatasetV3_2_090180270_controlnet":
        return ObjaverseDatasetV3_2_090180270_controlnet(configs, data_type=data_type)