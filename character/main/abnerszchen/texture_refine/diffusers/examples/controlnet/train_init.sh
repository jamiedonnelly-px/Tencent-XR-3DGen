export MODEL_DIR="/aigc_cfs/model/stable-diffusion-v1-5"
output_dir="/aigc_cfs_3/sz/result/tex_control/init"

mkdir -p ${output_dir}
log_txt=${output_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$output_dir \
 --dataset_name="/aigc_cfs/sz/dataset/fusing___fill50k/" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo"