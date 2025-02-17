current_script_path="$(realpath "$0")"
current_script_dir="$(dirname "$current_script_path")"
parent_dir="$(dirname "$current_script_dir")"
in_pose_json=${parent_dir}/data/cams/cam_parameters_select.json

# in_raw_json=/apdcephfs_cq3/share_2909871/3dAsset_artcenter/alldata_1204.json
in_raw_json=/apdcephfs_cq3/share_1615605/neoshang/code/rendering_free_onetri/savedir/test_128_v13/alldata_1113_right_v3.json
# in_gt_json=/aigc_cfs/sz/result/tex/render_two_list_in_first_2k/only_emission.json
# in_gt_json=/aigc_cfs_2/sz/dataset/tex/second_render_2k/only_emission.json
in_gt_json=/apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug/only_emission.json

in_est_dir=$1
out_dir=$2

cd ${current_script_dir}
python pre_find_gt_mesh.py ${in_raw_json} ${in_est_dir} ${out_dir}
python pre_batch_render_est.py ${out_dir}/est_objs.txt ${in_pose_json} ${out_dir}

#  wait senbo render gt.. then set in_gt_json to run
python pre_make_dataset_json.py ${in_gt_json} ${out_dir} ${out_dir}
python pre_split_train_test.py ${out_dir}/tex_refine.json ${out_dir}

# use blender render ${out_dir}/proc_data_paths.txt and ${out_dir}/mesh_paths.txt

# /apdcephfs_cq3/share_2909871/shenzhou/proj/DiffusionSDF/config/sz_diffusion_4096_v0_test/recon2023-12-06-15:46:04