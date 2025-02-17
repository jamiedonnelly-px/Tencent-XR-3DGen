
codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

# "tex_gen.json"

in_sd_path="/aigc_cfs_gdp/model/stable-diffusion-xl-base-1.0"
pretrained_vae_model_name_or_path="/aigc_cfs_gdp/model/sdxl-vae-fp16-fix"
ip_adapter_model_path="/aigc_cfs_gdp/model/IP-Adapter"

data_type="mcwy"
if [ ${data_type} = "ready" ]; then
    in_model_path="/aigc_cfs_gdp/sz/runtime_model/xl_mcwy2_manual/g4_pre_pos_4class_blip_1e-5/checkpoint-2000"
    in_dataset_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_top_right_test.json"
    model_key="control_ready"
elif [ ${data_type}  = "mcwy" ]; then
    in_model_path="/aigc_cfs_gdp/sz/runtime_model/xl_mcwy2_manual/g4_pre_pos_4class_blip_1e-5/checkpoint-2000"
    in_dataset_json="/aigc_cfs_3/layer_tex/readyplayerme/image_caption_top_right_test.json"
    model_key="uv_mcwy"
elif [ ${data_type}  = "human" ]; then
    in_model_path="/aigc_cfs_3/sz/result/tex_control_2024/xl_human/g4_pos_designonly_llava_5e-5"
    in_dataset_json="/aigc_cfs_9/sz/layer_tex9/human/design_llava_test.json"
    # in_model_path="/aigc_cfs_3/sz/result/tex_control_2024/xl_human/g4_pre_right_pos_4class_5e-5"
    # in_dataset_json="/aigc_cfs_9/sz/layer_tex9/human/right_test.json"
    model_key="uv_mcwy"
else
    echo "Invalid data_type: $data_type"
    exit 1
fi


infer_cnt=1
# out_dir=${in_model_path}/new_objs_${infer_cnt}_test
out_dir=/aigc_cfs_gdp/sz/batch_1012/texuv_test_new_${infer_cnt}

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


cmd="python run_texuv.py ${in_model_path} ${in_sd_path} ${pretrained_vae_model_name_or_path} ${ip_adapter_model_path} ${in_dataset_json}  ${out_dir}  --infer_cnt ${infer_cnt} --model_key ${model_key}"
echo ${cmd}
eval ${cmd}
