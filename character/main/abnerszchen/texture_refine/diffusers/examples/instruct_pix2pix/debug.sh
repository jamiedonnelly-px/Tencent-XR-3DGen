log_txt=/apdcephfs_cq3/share_2909871/shenzhou/result/pix2pix/exam_debug.txt
exec > >(tee ${log_txt}) 2>&1

cd /apdcephfs_cq3/share_2909871/shenzhou/proj/texture_refine/diffusers/examples/instruct_pix2pix/

export MODEL_NAME="/apdcephfs_cq8/share_2909871/shenzhou/model/stable-diffusion-v1-5"
export DATASET_ID="/apdcephfs_cq8/share_2909871/shenzhou/data/pixel/fusing___instructpix2pix-1000-samples"
output_dir=/apdcephfs_cq8/share_2909871/shenzhou/result/tex/debug

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --output_dir=$output_dir \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --random_flip \
    --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42
    


