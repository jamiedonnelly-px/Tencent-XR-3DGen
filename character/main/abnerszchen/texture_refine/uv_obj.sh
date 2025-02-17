
codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

in_sd_path="/aigc_cfs/model/stable-diffusion-v1-5"
ip_adapter_model_path="/aigc_cfs/model/IP-Adapter"

data_type="human"
if [ ${data_type} = "mcwy" ]; then
    # in_model_path="/aigc_cfs_3/sz/result/tex_control_2024/mcwy2/mask_pre_short"
    in_model_path="/aigc_cfs_3/sz/result/tex_control_2024/mcwy2/newdataset_pre_long"
    # in_dataset_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/right_test.json"    
    in_dataset_json="/aigc_cfs_3/layer_tex/mcwy_2/2024/infer.json"
elif [ ${data_type}  = "ready" ]; then
    in_model_path="/aigc_cfs_3/sz/result/tex_control_2024/ready/iblip_mask_pre_short"
    in_dataset_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_done_right_test.json"   
elif [ ${data_type}  = "human" ]; then
    in_model_path="/aigc_cfs_3/sz/result/tex_control_2024/human/design_llava_5e-5"
    in_dataset_json="/aigc_cfs_9/sz/layer_tex9/human/design_llava_test.json"   
else
    echo "Invalid data_type: $data_type"
    exit 1
fi

infer_cnt=10
out_dir=${in_model_path}/new_objs_${infer_cnt}_test

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


cmd="python run_texcontrol.py ${in_model_path} ${in_sd_path} ${ip_adapter_model_path} ${in_dataset_json}  ${out_dir}  --infer_cnt ${infer_cnt}"
echo ${cmd}
eval ${cmd}
