
data_type="lowpoly"
if [ ${data_type} = "lowpoly" ]; then
    model_path="/aigc_cfs_3/sz/result/tex_img_uv/g1/lowpoly_debug"
    in_json="/aigc_cfs/sz/data/tex/lowpoly/add_imgs_test.json"
    # in_json="/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/valid_right_test.json"
elif [ ${data_type}  = "mcwy" ]; then
    echo "Invalid data_type: $data_type"
    exit 1
else
    echo "Invalid data_type: $data_type"
    exit 1
fi

infer_cnt=20
# out_dir=${model_path}/infer_${infer_cnt}_wo_ip
# extra_args=""
out_dir=${model_path}/infer_${infer_cnt}_ip
extra_args="--test_ip"

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python infer_img_uv.py ${model_path} ${in_json} ${out_dir} --infer_cnt ${infer_cnt} ${extra_args}