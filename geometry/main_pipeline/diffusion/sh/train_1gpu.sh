
#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

# #### train image to 3D with 1 image condition
# CUDA_VISIBLE_DEVICES=0 \
# accelerate launch --num_processes 1 --mixed_precision "bf16" --main_process_port 29601 \
#     scripts/train_mmdit_image23D_flow_1view.py \
#     --output_dir="configs/1view_gray_2048_flow" \
#     --pretrained_model_name_or_path="configs/1view_gray_2048_flow" \
#     --validation_images_dir="/data/validation/images" \
#     --tracker_project_name "1view_gray_2048_flow" \
#     --use_ema \
#     --do_classifier_free_guidance \
#     --train_batch_size=2 --gradient_accumulation_steps=1  --gradient_checkpointing \
#     --num_train_epochs=100 \
#     --checkpointing_steps=2000 --checkpoints_total_limit=20 \
#     --lr_scheduler "cosine" --learning_rate=1e-4  --lr_warmup_steps=1000 --lr_num_cycles=4 \
#     --max_grad_norm=1 \
#     --dataloader_num_workers 8 \
#     --mixed_precision="bf16" \
#     --drop_condition_prob 0.1 \
#     --report_to="tensorboard"
# wait

### train image to 3D with multi images condition
CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_processes 1 --mixed_precision "bf16" --main_process_port 29601 \
    scripts/train_mmdit_image23D_flow_4views.py \
    --output_dir="configs/4view_gray_2048_flow" \
    --pretrained_model_name_or_path="configs/4view_gray_2048_flow" \
    --validation_images_dir="/data/validation/images" \
    --tracker_project_name "4view_gray_2048_flow" \
    --use_ema \
    --do_classifier_free_guidance \
    --train_batch_size=2 --gradient_accumulation_steps=1  --gradient_checkpointing \
    --num_train_epochs=100 \
    --checkpointing_steps=2000 --checkpoints_total_limit=20 \
    --lr_scheduler "cosine" --learning_rate=1e-4  --lr_warmup_steps=1000 --lr_num_cycles=4 \
    --max_grad_norm=1 \
    --dataloader_num_workers 8 \
    --mixed_precision="bf16" \
    --drop_condition_prob 0.1 \
    --report_to="tensorboard"
wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

echo "E2E Training Duration sec : $e2e_time"