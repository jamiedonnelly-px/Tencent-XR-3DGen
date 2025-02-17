
in_json="/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/right.json"
out_dir="/aigc_cfs_3/layer_tex/uv_datasets/mcwy2_manual_4class_pos/right_blip_0416"
out_910b_dir="/data5/sz/uv_datasets/mcwy2_manual_4class_pos/right_blip_0416"

# in_json="/aigc_cfs_9/sz/layer_tex9/human/right.json"
# out_dir="/aigc_cfs_3/layer_tex/uv_datasets/human_pos/all"
# out_910b_dir="/data5/sz/uv_datasets/human_pos/all"

# in_json="/aigc_cfs_3/layer_tex/readyplayerme/right_llava_3class.json"
# out_dir="/aigc_cfs_3/layer_tex/uv_datasets/ready_pos/llava_3class"
# out_910b_dir="/data5/sz/uv_datasets/ready_pos/llava_3class"

python make_uv_json.py  ${in_json}  ${out_dir} --out_910b_dir ${out_910b_dir}
