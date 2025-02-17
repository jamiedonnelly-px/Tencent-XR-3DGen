
export MODEL_NAME="/aigc_cfs/model/stable-diffusion-v1-5"
train_cnt="2k"
if [ ${train_cnt} = "300" ]; then
    export DATASET_JSON="/aigc_cfs/sz/data/tex/weapon_srender/sample300_tex_creator_train.json"
    export DATASET_VAL_JSON="/aigc_cfs/sz/data/tex/weapon_srender/sample300_tex_creator_val.json"
    export DATASET_TEST_JSON="/aigc_cfs/sz/data/tex/weapon_srender/sample300_tex_creator_test.json"
    max_train_steps=8000
    checkpointing_steps=4000
elif [ ${train_cnt}  = "2k" ]; then
    export DATASET_JSON="/aigc_cfs_3/sz/data/tex/weapon_srender/emission/tex_creator_train.json"
    export DATASET_VAL_JSON="/aigc_cfs_3/sz/data/tex/weapon_srender/emission/tex_creator_val.json"
    export DATASET_TEST_JSON="/aigc_cfs_3/sz/data/tex/weapon_srender/emission/tex_creator_test.json"
    max_train_steps=6000
    checkpointing_steps=2000
else
    echo "Invalid argument: $train_cnt"
    exit 1
fi

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

train_batch_size=16
gradient_accumulation_steps=2
validation_epochs=2
noise_scheduler="ddpm"

output_dir=/aigc_cfs_3/sz/result/tex_creator/obja_emission_srender_argum/g8/first_${train_cnt}_b${train_batch_size}a${gradient_accumulation_steps}_ns${noise_scheduler}
mkdir -p ${output_dir}
log_txt=${output_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

accelerate launch --mixed_precision="fp16" train_tex_creator.py \
    --resume_from_checkpoint=latest \
    --dataset_argum \
    --dataloader_num_workers=32 \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --noise_scheduler=$noise_scheduler \
    --dataset_json=$DATASET_JSON \
    --dataset_val_json=$DATASET_VAL_JSON \
    --dataset_test_json=$DATASET_TEST_JSON \
    --output_dir=$output_dir \
    --validation_epochs=$validation_epochs \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 \
    --train_batch_size=${train_batch_size} --gradient_accumulation_steps=${gradient_accumulation_steps} --gradient_checkpointing \
    --max_train_steps=${max_train_steps} \
    --checkpointing_steps=${checkpointing_steps} --checkpoints_total_limit=10 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42
    

#  --random_flip


