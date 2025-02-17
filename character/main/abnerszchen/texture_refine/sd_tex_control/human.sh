MODEL_DIR="/aigc_cfs/model/stable-diffusion-v1-5"
# controlnet_model_name_or_path="/aigc_cfs/model/control_v11f1p_sd15_depth"
controlnet_model_name_or_path=""

output_dir="/aigc_cfs_3/sz/result/tex_control_2024/human/design_iblip_5e-5"
dataset_json="/aigc_cfs_9/sz/layer_tex9/human/design_iblip_train.json"
dataset_val_json="/aigc_cfs_9/sz/layer_tex9/human/design_iblip_test.json"
dataset_test_json=""
num_train_epochs=40
# dataset_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_done_right_train.json"
# dataset_val_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_done_right_test.json"
# dataset_test_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_done_right_infer.json"
# num_train_epochs=600

# output_dir="/aigc_cfs_3/sz/result/tex_control_2024/mcwy2/scratch_prob0.1_mask_gt"
# dataset_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/right_train.json"
# dataset_val_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/right_test.json"
# dataset_test_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/right_infer.json"
# num_train_epochs=300

# max_train_steps=10000
checkpointing_steps=1000
validation_steps=500
train_batch_size=16
gradient_accumulation_steps=4
proportion_empty_prompts=0.1
learning_rate=5e-5

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

current_time=$(date +"%Y-%m-%d_%H-%M")
mkdir -p ${output_dir}
log_txt=${output_dir}/log_${current_time}.txt
exec > >(tee ${log_txt}) 2>&1

accelerate launch --mixed_precision="fp16" --multi_gpu train_tex_control.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$controlnet_model_name_or_path \
 --dataset_argum \
 --num_train_epochs ${num_train_epochs}\
 --output_dir=$output_dir \
 --dataset_json=${dataset_json} \
 --dataset_val_json=${dataset_val_json} \
 --dataset_test_json=${dataset_test_json} \
 --validation_steps=${validation_steps} \
 --checkpointing_steps=${checkpointing_steps} \
 --train_batch_size=${train_batch_size} \
 --gradient_accumulation_steps=${gradient_accumulation_steps} \
 --proportion_empty_prompts=${proportion_empty_prompts} \
 --resolution=512 \
 --learning_rate=${learning_rate} \
 --dataloader_num_workers=8 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --seed 42

#  --dataset_mask_gt \

#  --dataset_name="/aigc_cfs/sz/dataset/fusing___fill50k/" \
