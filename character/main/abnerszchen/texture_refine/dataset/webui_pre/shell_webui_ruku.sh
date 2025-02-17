codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

current_time=$(date +"%Y-%m-%d_%H-%M")
export in_source_json=/aigc_cfs_11/Asset/active_list/layered_data/correct_version/20240925_gdp.json
export in_flatten_json=/aigc_cfs_2/sz/proj/tex_cq/configs/web_0711/web_flatten_gdp_manual_fixuv.json
export out_dir=/aigc_cfs_gdp/Asset/clothes/process_sz/web_0925
mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

python step1_merge_flatten_to_source.py --in_source_json ${in_source_json} \
 --in_flatten_json ${in_flatten_json} 2>&1 \
 --out_json ${out_dir}/s1_merge_flatten_gdp.json 2>&1 | tee ${out_dir}/log_step1.txt

python step2_3_gen_uv.py --in_new_json ${out_dir}/s1_merge_flatten_gdp_new_need_check.json \
 --out_dir ${out_dir} 2>&1 | tee ${out_dir}/log_step2_3.txt

python step4_gen_new_json.py --in_setok_json ${out_dir}/s1_merge_flatten_gdp_setok.json \
 --in_gen_uv_json ${out_dir}/generate_uv_done.json \
 --out_dir ${out_dir} 2>&1 | tee ${out_dir}/log_step4.txt

python step5_check_render.py --in_json ${out_dir}/final_merge.json \
 --out_dir ${out_dir} 2>&1 | tee ${out_dir}/log_step5.txt

python step6_flatten_to_web_json.py --in_standard_json ${out_dir}/final_ok.json \
 --out_json ${out_dir}/web_flatten_gdp.json

ls ${out_dir}/final_ok.json
ls ${out_dir}/web_flatten_gdp.json


#### then manual cp to configs/*data/