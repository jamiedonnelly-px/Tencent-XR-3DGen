log_txt=/apdcephfs_cq3/share_2909871/shenzhou/result/pix2pix/exam.txt
exec > >(tee ${log_txt}) 2>&1
codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}
export MODEL_NAME="/aigc_cfs/model/stable-diffusion-v1-5"
export DATASET_ID="/aigc_cfs/sz/dataset/fusing___instructpix2pix-1000-samples"
output_dir=/apdcephfs_cq8/share_2909871/shenzhou/result/tex/init

train_batch_size=2
gradient_accumulation_steps=2

accelerate launch --mixed_precision="fp16" --multi_gpu train_tex_refine.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset_name=$DATASET_ID  \
 --output_dir=$output_dir \
 --use_ema \
 --enable_xformers_memory_efficient_attention \
 --resolution=512 --random_flip \
 --train_batch_size=${train_batch_size} --gradient_accumulation_steps=${gradient_accumulation_steps} --gradient_checkpointing \
 --max_train_steps=15000 \
 --checkpointing_steps=5000 --checkpoints_total_limit=1 \
 --learning_rate=5e-05 --lr_warmup_steps=0 \
 --conditioning_dropout_prob=0.05 \
 --mixed_precision=fp16 \
 --seed=42 


