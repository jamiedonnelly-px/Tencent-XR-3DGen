codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

current_time=$(date +"%Y-%m-%d_%H-%M")
output_dir=/aigc_cfs/sz/result/tex_control/dataset_log/${current_time}
mkdir -p ${output_dir}
log_txt=${output_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

# readyplayerme
render_dir='/aigc_cfs/layer_avatar_data/readplayerMe/render/render_data'
python cmds_ready_glb_obj.py /aigc_cfs/layer_avatar_data/readplayerMe/top_bottom_footwear_glb.txt /aigc_cfs/layer_avatar_data/readplayerMe/objs_three
python ../../scripts/utils_pool_cmds.py /aigc_cfs/layer_avatar_data/readplayerMe/objs_three/glb_to_obj_cmds.txt
python cmds_ready_uv.py /aigc_cfs/layer_avatar_data/readplayerMe/objs_three
python ../../scripts/utils_pool_cmds.py /aigc_cfs/layer_avatar_data/readplayerMe/objs_three/generate_uv_cmds.txt
python json_readyme.py /aigc_cfs/layer_avatar_data/readplayerMe/objs_three
python ready_find_caption.py /aigc_cfs/layer_avatar_data/readplayerMe/objs_three/valid.json ${render_dir} /aigc_cfs/layer_avatar_data/readplayerMe/objs_three/valid_caption.json
python ../pre_split_train_test.py /aigc_cfs/layer_avatar_data/readplayerMe/objs_three/valid_caption.json /aigc_cfs/layer_avatar_data/readplayerMe/objs_three


# mcwy2
export source_json=/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/source_mcwy2_4class.json
export mcwy_out_dir="/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416"
mkdir -p ${mcwy_out_dir}
python mcwy_2_merge_mtl.py ${source_json} ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_2_merge_mtl.txt
python mcwy_3_generate_uv.py ${mcwy_out_dir}/merge_mtl_done.json ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_3_uv.txt
python mcwy_4_find_image.py ${mcwy_out_dir}/generate_uv_done.json ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_4_find_image.txt
python mcwy_5_image_caption_gpus.py ${mcwy_out_dir}/find_image_done.json ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_5_caption.txt
python mcwy_6_select_data.py ${mcwy_out_dir}/image_caption_done.json ${mcwy_out_dir}/right.json 2>&1 | tee ${mcwy_out_dir}/log_6_select.txt
python mcwy_6_select_data.py ${mcwy_out_dir}/image_caption_done.json ${mcwy_out_dir}/select_all.json --select_all 2>&1 | tee ${mcwy_out_dir}/log_6_select.txt
python ../pre_split_train_test.py ${mcwy_out_dir}/right.json ${mcwy_out_dir} --test_ratio 0.05
python ../pre_split_train_test.py ${mcwy_out_dir}/select_all.json ${mcwy_out_dir} --test_ratio 0.05

# human with multi gpus
export source_json=/aigc_cfs_9/sz/layer_tex9/human/source.json
export mcwy_out_dir="/aigc_cfs_9/sz/layer_tex9/human/"
mkdir -p ${mcwy_out_dir}
python mcwy_2_merge_mtl.py ${source_json} ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_2_merge_mtl.txt
python mcwy_3_generate_uv.py ${mcwy_out_dir}/merge_mtl_done.json ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_3_uv.txt
python mcwy_4_find_image.py ${mcwy_out_dir}/generate_uv_done.json ${mcwy_out_dir} --data_type human 2>&1 | tee ${mcwy_out_dir}/log_4_find_image.txt
python mcwy_5_image_caption_gpus.py ${mcwy_out_dir}/find_image_done.json ${mcwy_out_dir} --num_gpus 8 2>&1 | tee ${mcwy_out_dir}/log_5_caption.txt
python mcwy_6_select_data.py ${mcwy_out_dir}/image_caption_done.json ${mcwy_out_dir}/right.json 2>&1 | tee ${mcwy_out_dir}/log_6_select.txt
python ../pre_split_train_test.py ${mcwy_out_dir}/right.json ${mcwy_out_dir}


# # mcwy2
# # /aigc_cfs/Asset/lists/mcwy_part2_20240204.json
# source_json="/aigc_cfs/Asset/designcenter/clothes/mcwy_data.json"
# mcwy_out_dir="/aigc_cfs_3/layer_tex/mcwy_2/objs_layer"
# python mcwy_1_fbx_to_obj.py ${source_json} ${mcwy_out_dir}
# python mcwy_2_merge_mtl.py ${mcwy_out_dir}/cvt_obj_done.json ${mcwy_out_dir}
# python mcwy_3_generate_uv.py ${mcwy_out_dir}/merge_mtl_done.json ${mcwy_out_dir} 2>&1 | tee ${mcwy_out_dir}/log_3_uv.txt

# /aigc_cfs/sz/software/blender-3.6.2-linux-x64/blender -b -P blender_addon_install.py -- --addon_path /aigc_cfs/sz/software/material-combiner-addon-master.zip

# python cmds_mcwy_fbx_obj.py /aigc_cfs/Asset/designcenter/clothes/mcwy_data.json /aigc_cfs/layer_avatar_data/mcwy_2/objs_three
# python ../../scripts/utils_pool_cmds.py /aigc_cfs/layer_avatar_data/mcwy_2/objs_three/fbx_to_obj_cmds.txt
# python cmds_mcwy_uv.py /aigc_cfs/layer_avatar_data/mcwy_2/objs_three
# python ../../scripts/utils_pool_cmds.py /aigc_cfs/layer_avatar_data/mcwy_2/objs_three/generate_uv_cmds.txt


