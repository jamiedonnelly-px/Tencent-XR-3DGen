current_script_path="$(realpath "$0")"
current_script_dir="$(dirname "$current_script_path")"
parent_dir="$(dirname "$current_script_dir")"

in_raw_json=/aigc_cfs/neoshang/data/json_for_traintest/objaverse/latent_geotri_Transformer_v20_128_obj_20231219_neo_20231219_add_condition_sort_images.json


in_est_dir=$1
in_est_test_dir=$2
out_dir=$3

cd ${current_script_dir}
mkdir -p ${out_dir}

log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


# ls ${in_est_dir}/*/*/0000/mesh.obj > ${out_dir}/est_objs.txt
# ls ${in_est_test_dir}/*/*/0000/mesh.obj >> ${out_dir}/est_objs.txt

### just use gt depth
python direct_make_gtD_json.py ${in_raw_json} ${out_dir}/est_objs.txt ${out_dir} 


### old version with render diffusion obj depth
# python render_est_lrm.py ${in_raw_json} ${out_dir}/est_objs.txt ${out_dir} 

# python make_split_json_lrm.py ${in_raw_json} ${out_dir}