
data_type="human"
if [ ${data_type} = "ready" ]; then
    model_path="/aigc_cfs_3/sz/result/tex_control_2024/ready/iblip_mask_top"
    in_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_top_right_infer.json"
    # in_json="/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_test.json"
elif [ ${data_type}  = "mcwy" ]; then
    model_path="/aigc_cfs_3/sz/result/tex_control_2024/mcwy2/pos/g8_pre_prob0.1"
    # model_path="/aigc_cfs_3/sz/result/tex_control_2024/mcwy2/pos/g8_pre_prob0.1_lla_long"
    in_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/infer.json"
    # in_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/right.json"
    # in_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/caption_lla_right_test.json"
    # in_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/caption_lla_right.json"
elif [ ${data_type}  = "human" ]; then
    model_path="/aigc_cfs_3/sz/result/tex_control_2024/human/pre_5e-5"
    # model_path="/aigc_cfs_3/sz/result/tex_control_2024/mcwy2/pos/g8_pre_prob0.1_lla_long"
    in_json="/aigc_cfs_9/sz/layer_tex9/human/right_test.json"
    # in_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/right.json"
    # in_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/caption_lla_right_test.json"
    # in_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/caption_lla_right.json"
else
    echo "Invalid data_type: $data_type"
    exit 1
fi


infer_cnt=20
add_params=""
out_dir=${model_path}/batch_infer_all_${infer_cnt}

# add_params="--test_ip"
# out_dir=${model_path}/infer_ip_${infer_cnt}

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python infer_control.py ${model_path} ${in_json} ${out_dir} --infer_cnt ${infer_cnt} ${add_params}