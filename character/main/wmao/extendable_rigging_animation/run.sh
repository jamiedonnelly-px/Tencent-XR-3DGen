#!/bin/bash

MODELS=("PointTransformerSeg38" "PointTransformerSeg50" "PointTransformerSeg26")
NUM_PTS=(4096 2048)

for NP in "${NUM_PTS[@]}"; do
    for MO in "${MODELS[@]}"; do
        echo "Processing: $MO with $NP points"
        ((NP = NP))
        CUDA_LAUNCH_BLOCKING=1 python train_point_transformer_contrast_new.py --model "$MO" --npoint "$NP" --rot_aug --batch_size 16
    done
done



# # MODELS=("PointTransformerSeg26" "PointTransformerSeg38" "PointTransformerSeg50")
# # NUM_PTS=(2048 4096)
# MODELS=("PointTransformerSeg38")
# NUM_PTS=(2048)

# for MO in "${MODELS[@]}"; do
#     for NP in "${NUM_PTS[@]}"; do
#         echo "Processing: $MO with $NP points"
#         ((NP = NP))
#         python train_partseg_point_transformer_new.py --model "$MO" --npoint "$NP" --dataset_name "$DATASET" --label_dir "$LABDIR"
#     done
# done