codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

current_time=$(date +"%Y-%m-%d_%H-%M")
export in_raw_source_json=/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/20241010_daz_decimate_add_ct.json
export in_flatten_json=/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/web_flatten_gdp_before_pre.json
export out_dir=/aigc_cfs_gdp/Asset/clothes/process_sz/web_1010/pre
mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


# python ../control_pre/mtl_manual/temp_web_to_standard.py ${in_flatten_json} ${out_dir}/web_flatten_gdp_before_pre.json


python step2_3_gen_uv.py --in_new_json ${out_dir}/web_flatten_gdp_before_pre.json \
 --out_dir ${out_dir} 2>&1 | tee ${out_dir}/log_step2_3.txt


python step7_check_obj_one_kd.py --in_raw_source_json ${in_raw_source_json} \
 --in_uv_json ${out_dir}/generate_uv_done.json \
 --out_dir ${out_dir}/replace 2>&1 | tee ${out_dir}/log_step7.txt

python prerender_step8_use_mergemtl_mesh.py 
--in_source_json ${out_dir}/replace/final_generate_uv_done.json
--out_json ${out_dir}/replace/mesh_single_kd_source.json