
data_type="mcwy"
if [ ${data_type} = "ready" ]; then
    model_path="/aigc_cfs_3/sz/result/tex_control_2024/xl_ready/g4_pre_xyz_fixvae"
    in_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_top_right_test.json"
    # in_json="/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_test.json"
elif [ ${data_type}  = "mcwy" ]; then
    # model_path="/aigc_cfs_3/sz/result/tex_control_2024/mcwy2/pos/g8_pre_prob0.1"
    # model_path="/aigc_cfs_3/sz/result/tex_control_2024/xl_mcwy2/g4_pre_right_pos_4class"
    model_path="/aigc_cfs_3/sz/result/tex_control_2024/xl_mcwy2_manual/g4_small_pre_pos_4class_blip_1e-5/checkpoint-600/controlnet"
    # model_path="/aigc_cfs_3/sz/result/tex_control_2024/xl_mcwy2_manual/g4_pre_pos_4class_blip_1e-5/checkpoint-2000/controlnet"
    in_json="/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/right_test.json"
else
    echo "Invalid data_type: $data_type"
    exit 1
fi


infer_cnt=10
add_params=""
out_dir=${model_path}/batch_test_${infer_cnt}_raw

# infer_cnt=2
# add_params="--test_ip"
# out_dir=${model_path}/infer_ip_rawsche_${infer_cnt}

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python infer_xl_control.py ${model_path} ${in_json} ${out_dir} --infer_cnt ${infer_cnt} ${add_params}
# python infer_xl_control_img2img.py ${model_path} ${in_json} ${out_dir} --infer_cnt ${infer_cnt} ${add_params}