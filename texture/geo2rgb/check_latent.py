import numpy as np
import os
import json
from tqdm import tqdm

def weight_acc(num_list, prob_list):
    print(len(num_list))
    print(len(prob_list))
    assert(len(num_list) == len(prob_list))
    all_num = np.sum(np.array(num_list))
    prob_weight = 0
    for num, prob in zip(num_list, prob_list):
        prob_weight += prob * (num / all_num)
    return prob_weight

data_json_path = "/aigc_cfs_2/neoshang/data/data_list/alldata_20240204_neo_v20240407_caption_910b_nobuilding_train.json"
stats_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/text_to_3d/910b_all_v0.0.0"
class_name_list = []
prob = 1.0

# latent_dir = "/aigc_cfs/weixuan/data/latent_map_twotri_Transformer_v1_128_vq_obj_20231214"
# stats_save_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdfcolor_objaverse_vq_v0.0.0"

# latent_dir = "/apdcephfs_cq3/share_1615605/weixuansun/data/latent_vector_transformer_v10_20231122/"
# stats_save_dir = "/aigc_cfs_2/neoshang/code/Diffusion-SDF/store/config/stage2_diffusion_cond_vector_sdfcolor_4096_ema_v0.4.0"

# latent_dir = "/aigc_cfs_2/neoshang/code/Diffusion-SDF/store/config/stage1_vae_dmtet_vector_v2/latent_train1287"
# stats_save_dir = "/aigc_cfs_2/neoshang/code/Diffusion-SDF/store/config/stage1_vae_dmtet_vector_v2"

# latent_dir = "/aigc_cfs_2/neoshang/code/Diffusion-SDF/store/config/stage1_vae_dmtet_vector_v3/latent_train695"
# stats_save_dir = "/aigc_cfs_2/neoshang/code/Diffusion-SDF/store/config/stage1_vae_dmtet_vector_v3"

# latent_dir = "/aigc_cfs_2/neoshang/code/Diffusion-SDF/store/config/stage1_vae_dmtet_vector_objaverse_v1.2.2/latent_train171"
# stats_save_dir = "/aigc_cfs_2/neoshang/code/Diffusion-SDF/store/config/stage2_diff_cond_vector_dmtet_objaverse_2048_v0.1.0"

os.makedirs(stats_save_dir, exist_ok=True)

with open(data_json_path, 'r') as fr:
    json_dict = json.load(fr)
data_dict = json_dict["data"]



min_list = []
max_list = []
mean_list = []
num_list = []
std_list = []
print_shape = False

for class_name, class_dict in data_dict.items():
    if (len(class_name_list) > 0) and (class_name not in class_name_list):
        continue
    min_num = 9999999
    max_num = -9999999
    latent_list = []
    num = 0
    for objname, objdict in tqdm(class_dict.items()):
        if np.random.random() > prob:
            continue
        if "latent" not in objdict:
            continue
        try:
            latent = np.load(objdict["latent"]).squeeze()
        except:
            continue
        if len(latent.shape) < 2:
            if latent.shape[-1] > 1024:
                latent_length = latent.shape[-1]
                latent = latent[..., :int(latent_length // 2)]
        else:
            latent_length = latent.shape[0]
            latent = latent[:int(latent_length // 2), ...]
        if not print_shape:
            print("latent_shape: ", latent.shape)
            print_shape = True

        if min_num > latent.min():
            min_num = latent.min()
        if max_num < latent.max():
            max_num = latent.max()
        if num < 5000:
            latent_list.append(latent)
        num += 1
    if num > 1:
        latent_array = np.concatenate(latent_list, axis=-1)
        std_list.append(np.std(latent_array))
        mean_list.append(np.mean(latent_array))
        min_list.append(min_num)
        max_list.append(max_num)
        num_list.append(num + 1)
        print(f"{class_name}: min_num: {min_num}, max_num: {max_num}, std: {np.std(latent_array)}, mean: {np.mean(latent_array)}")
min_all = weight_acc(num_list, min_list)
max_all = weight_acc(num_list, max_list)
mean_all = weight_acc(num_list, mean_list)
std_all = weight_acc(num_list, std_list)

mean_min_max = (min_all + max_all) / 2
range = max_all - min_all
range = range / 4

np.save(f'{stats_save_dir}/range', range)
np.save(f'{stats_save_dir}/mean', mean_min_max)

print(f"1/std_all: {1/std_all}")
print(f"min_all: {min_all}, max_all: {max_all}, mean_min_max: {mean_min_max}, mean_all: {mean_all}, range: {range}, std_all: {std_all}")
print(f"rescale min: {(min_all - mean_min_max) / (range / 2)}, recale max: {(max_all - mean_min_max) / (range / 2)}")
