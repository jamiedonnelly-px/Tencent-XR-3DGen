source /root/anaconda3/bin/activate base

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

# config_path=$1
# accelerate launch --config_file $1 \
# accelerate launch $DISTRIBUTED_ARGS \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# accelerate launch --num_processes 8 --main_process_port 36698 \
# accelerate launch --multi_gpu --main_process_ip **** --machine_rank 0 --num_processes 16 --main_process_port 52196 --num_machines 1 \
#accelerate launch --multi_gpu --num_processes=16 --num_machines=1 --machine_rank=0 --main_process_ip=**** --main_process_port=52196 \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_processes 1 --main_process_port 52268 \
    examples/zero123plus/train_image_to_mv_step1_v22_fp16_1cond_controlnet.py \
    --output_dir="configs/zero123plus/zero123plus_v29_1cond_6views_090180270_controlnet" \
    --pretrained_model_name_or_path="/aigc_cfs/xibinsong/models/zero123plus_v29_1cond_6views_090180270" \
    --pretrained_controlnet_model_name_or_path="/aigc_cfs/model/controlnet-zp11-depth-v1" \
    --empty_prompt_embedding_path="/aigc_cfs/xibinsong/models/empty_prompt_embedding.pt" \
    --validation_images_dir="/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation" \
    --tracker_project_name "zero123plus_v29_1cond_6views_090180270" \
    --use_ema \
    --do_classifier_free_guidance \
    --prediction_type "v_prediction" \
    --train_batch_size=1 --gradient_accumulation_steps=6 --gradient_checkpointing \
    --num_train_epochs=6 \
    --validation_epochs=1 \
    --checkpointing_steps=3000 --checkpoints_total_limit=200 \
    --lr_scheduler "cosine" --learning_rate=5e-05 --lr_warmup_steps=1000 \
    --max_grad_norm=1 \
    --dataloader_num_workers 8 \
    --mixed_precision="fp16" \
    --snr_gamma 5.0 \
    --drop_condition_prob 0.1 \
    --report_to="tensorboard" \
    --resume_from_checkpoint "latest"


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

echo "E2E Training Duration sec : $e2e_time"
