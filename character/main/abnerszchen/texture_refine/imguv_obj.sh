
codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

in_sd_path="/aigc_cfs/model/stable-diffusion-v1-5"
ip_adapter_model_path="/aigc_cfs/model/IP-Adapter"

data_type="mcwy"
if [ ${data_type} = "mcwy" ]; then
    in_model_path="/aigc_cfs_3/sz/result/tex_img_uv/g8/mcwy2_img"
    in_dataset_json="/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_cap_img_test.json"
elif [ ${data_type}  = "lowpoly" ]; then
    in_model_path="/aigc_cfs_3/sz/result/tex_img_uv/g1/lowpoly_debug"
    in_dataset_json="/aigc_cfs/sz/data/tex/lowpoly/add_imgs_test.json"
else
    echo "Invalid data_type: $data_type"
    exit 1
fi

 
infer_cnt=10
out_dir=${in_model_path}/uv_objs_${infer_cnt}_test_${data_type}_text

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


cmd="python run_teximguv.py ${in_model_path} ${in_sd_path} ${ip_adapter_model_path} ${in_dataset_json}  ${out_dir}  --infer_cnt ${infer_cnt}"
echo ${cmd}
eval ${cmd}
