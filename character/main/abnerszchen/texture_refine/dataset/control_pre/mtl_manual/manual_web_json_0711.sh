python temp_web_json_use_manual.py /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_0507.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json

python temp_web_to_standard.py /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/gdp_manual_standard.json

### replace pro...TODO

python ../mcwy_3_generate_uv.py /aigc_cfs_gdp/layer_tex/20240711_gdp/gdp_manual_standard.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/all_uv

python bad_uv_pack.py /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/all_uv/generate_uv_failed_cmds.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/bad_uv_pack


#### manual mv / fix uv

python manual_uv_replace_in_web.py /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/manual_move_uv \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual_fixuv.json


#### check again
cp /aigc_cfs_gdp/layer_tex/20240711_gdp/all_uv/generate_uv_failed_cmds.json /aigc_cfs_gdp/layer_tex/20240711_gdp/all_uv/bak_generate_uv_failed_cmds.json
python temp_web_to_standard.py /aigc_cfs_gdp/layer_tex/20240711_gdp/web_flatten_gdp_manual_fixuv.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/gdp_manual_fixuv_standard.json

### replace pro...TODO
##make sure all pass
python ../mcwy_3_generate_uv.py /aigc_cfs_gdp/layer_tex/20240711_gdp/gdp_manual_fixuv_standard_pro.json \
 /aigc_cfs_gdp/layer_tex/20240711_gdp/all_uv