in_json="/aigc_cfs_2/sz/proj/tex_cq/configs/web_0711/web_flatten_gdp_manual_fixuv.json"
out_root="/aigc_cfs_gdp/sz/result/test_tex_replace_0717"
test_cnt=-1
max_workers=2

mkdir -p ${out_root}
python test_all.py ${in_json} ${out_root} --test_cnt ${test_cnt} --max_workers ${max_workers} 2>&1 | tee ${out_root}/log.txt

python re_run_failed.py ${out_root}/failed.json ${out_root}/re_run_1 2>&1 | tee ${out_root}/re_failed.txt


