source /root/anaconda3/bin/activate base

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"


mkdir /root/.u2net
cp /aigc_cfs_2/neoshang/models/rembg/u2net.onnx /root/.u2net/u2net.onnx


CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_processes 1 --main_process_port 29601 \
    examples/zero123plus/train_image_to_mv_step1_v22_fp16_1cond.py \
    --output_dir="configs/zero123plus/zero123plus_v29_1cond_6views_090180270" \
    --pretrained_model_name_or_path="/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/zero123plus/zero123plus_v29_1cond_6views_090180270" \
    --validation_images_dir="/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation" \
    --tracker_project_name "zero123plus_v29_1cond_6views_090180270" \
    --use_ema \
    --do_classifier_free_guidance \
    --prediction_type "v_prediction" \
    --train_batch_size=10 --gradient_accumulation_steps=6 --gradient_checkpointing \
    --num_train_epochs=6 \
    --validation_epochs=1 \
    --checkpointing_steps=3000 --checkpoints_total_limit=20 \
    --lr_scheduler "cosine" --learning_rate=7e-05 --lr_warmup_steps=1000 \
    --max_grad_norm=1 \
    --dataloader_num_workers 8 \
    --mixed_precision="fp16" \
    --snr_gamma 5.0 \
    --drop_condition_prob 0.1 \
    --report_to="tensorboard"
    # --resume_from_checkpoint "latest"
wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

echo "E2E Training Duration sec : $e2e_time"