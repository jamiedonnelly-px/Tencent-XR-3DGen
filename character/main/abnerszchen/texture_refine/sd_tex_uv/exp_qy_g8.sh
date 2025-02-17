MODEL_DIR="/aigc_cfs/model/stable-diffusion-v1-5"
controlnet_model_name_or_path="/aigc_cfs/model/control_v11f1p_sd15_depth"

# output_dir="/aigc_cfs_3/sz/result/tex_uv/g8/ready_text_cap2_pos"
# dataset_json="/aigc_cfs/layer_avatar_data/readplayerMe/objs_three/blip/valid_caption2_train.json"
# dataset_val_json="/aigc_cfs/layer_avatar_data/readplayerMe/objs_three/blip/valid_caption2_test.json"
# dataset_test_json="/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_cap_test.json"

output_dir="/aigc_cfs_3/sz/result/tex_uv/g8/mcwy2_cap2_pos"
dataset_json="/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_cap_train.json"
dataset_val_json="/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_cap_test.json"
dataset_test_json="/aigc_cfs/layer_avatar_data/readplayerMe/objs_three/blip/valid_caption2_test.json"

max_train_steps=1000
checkpointing_steps=300
validation_steps=50
train_batch_size=16
gradient_accumulation_steps=2

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

current_time=$(date +"%Y-%m-%d_%H-%M")
mkdir -p ${output_dir}
log_txt=${output_dir}/log_${current_time}.txt
exec > >(tee ${log_txt}) 2>&1

accelerate launch --mixed_precision="fp16" --multi_gpu train_tex_uv.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$controlnet_model_name_or_path \
 --dataset_argum \
 --max_train_steps ${max_train_steps}\
 --output_dir=$output_dir \
 --dataset_json=${dataset_json} \
 --dataset_val_json=${dataset_val_json} \
 --dataset_test_json=${dataset_test_json} \
 --validation_steps=${validation_steps} \
 --checkpointing_steps=${checkpointing_steps} \
 --train_batch_size=${train_batch_size} \
 --gradient_accumulation_steps=${gradient_accumulation_steps} \
 --gradient_checkpointing \
 --resolution=512 \
 --learning_rate=1e-5 \
 --dataloader_num_workers=8 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --seed 42

#  --dataset_name="/aigc_cfs/sz/dataset/fusing___fill50k/" \
