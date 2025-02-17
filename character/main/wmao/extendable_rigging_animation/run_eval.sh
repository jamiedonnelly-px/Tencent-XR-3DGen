#!/bin/bash

# MODELS=("PointTransformerSeg50" "PointTransformerSeg38" "PointTransformerSeg26")
# NUM_PTS=(4096 2048)


# for NP in "${NUM_PTS[@]}"; do
#     for MO in "${MODELS[@]}"; do
#         echo "Processing: $MO with $NP points"
#         ((NP = NP))
#         python train_point_transformer_contrast_new.py --model "$MO" --npoint "$NP" --rot_aug --batch_size 32
#     done
# done

MO=PointTransformerSeg38
NP=4096
PT=/aigc_cfs_2/weimao/non-smalfit/output/SK_Dolphin/contrast_rot_aug_PointTransformerSeg38/2024-07-26_15-17-39
python train_point_transformer_contrast_new.py --model "$MO" --npoint "$NP" --rot_aug --batch_size 32 --pretrained_dir "$PT" --is_eval True