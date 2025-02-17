
export MODEL_NAME="/aigc_cfs/model/stable-diffusion-v1-5"
export DATASET_JSON="/aigc_cfs/sz/result/tex/first_2k/tex_refine_train.json"
export DATASET_TEST_JSON="/aigc_cfs/sz/result/tex/first_2k/tex_refine_test.json"
output_dir=/aigc_cfs/sz/result/tex/condi_g1/first_2k
mkdir -p ${output_dir}
log_txt=${output_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

max_train_steps=25000

accelerate launch --mixed_precision="fp16" train_tex_refine.py \
    --dataloader_num_workers=8 \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_json=$DATASET_JSON \
    --dataset_test_json=$DATASET_TEST_JSON \
    --output_dir=$output_dir \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 \
    --train_batch_size=1 --gradient_accumulation_steps=1 --gradient_checkpointing \
    --max_train_steps=${max_train_steps} \
    --checkpointing_steps=15000 --checkpoints_total_limit=2 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42
    

#  --random_flip


