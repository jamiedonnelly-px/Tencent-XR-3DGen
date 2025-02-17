# generate uv
python mtl_generate_uv_resized.py /aigc_cfs_3/layer_tex/20240709/20240709_ruku_ok_tex_daz.json /aigc_cfs_3/layer_tex/daz/direct_my_uv

# after manual
python mtl_make_manual_json.py /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_multi_kd_01_03_15_all/manual_multi_kd.txt /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_multi_kd_01_03_15_all/manual_multi_kd.json

python mtl_make_manual_json.py /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_one_kd_d/manual_multi_mtl_one_kd_d.txt /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_one_kd_d/manual_multi_mtl_one_kd_d.json --dir_type raw
python mtl_make_manual_json.py /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_multi_kd_d/manual_multi_kd_d.txt /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_multi_mtl_multi_kd_d/manual_multi_kd_d.json --dir_type raw
python mtl_make_manual_json.py /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_only_single_d/manual_only_single_d.txt /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/manual_only_single_d/manual_only_single_d.json --dir_type raw

python mtl_select_uv_resized_filter_ok.py /aigc_cfs_3/layer_tex/mcwy/direct_my_uv/generate_uv_done.json /aigc_cfs_3/layer_tex/mcwy/merge/mcwy2_mtl_pack/multi_kd_uv_1024_before_filter_maunal/filter_ok.txt /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/myuv_filter_ok_multi_kd.json

python mtl_select_bake_filter_ok.py /aigc_cfs_3/layer_tex/mcwy/merge/bake_info.json /aigc_cfs_3/layer_tex/mcwy/merge/mcwy2_mtl_pack/one_kd_uv_before_filter_manual/filter_ok_temp0406.txt /aigc_cfs_3/layer_tex/mcwy/merge/append_manual_data/bake_filter_ok_one_kd_temp0406.json

# merge after set txt
python mtl_append_to_source.py /aigc_cfs_3/layer_tex/mcwy/merge/source_mcwy2_4class.json /aigc_cfs_2/sz/proj/tex_cq/dataset/control_pre/mtl_manual/append_jsons.txt /aigc_cfs_3/layer_tex/mcwy/merge/preprocess_mcwy2_4class_0416.json


# make web
python mtl_make_web_json.py \
 /aigc_cfs_3/layer_tex/mcwy/merge/preprocess_mcwy2_4class_0416.json \
 /aigc_cfs_gdp/sz/data/layer_human/web_0507/layer_embedding_20240507_total.json \
 /aigc_cfs_gdp/sz/data/layer_human/web_0507/
 
# python mtl_make_web_json.py \
#  /aigc_cfs_3/layer_tex/mcwy/merge/preprocess_mcwy2_4class_0416.json \
#  /aigc_cfs_3/layer_tex/mcwy/merge/layer_embedding_20240403_total.json \
#  /aigc_cfs_3/layer_tex/mcwy/merge/web_0416