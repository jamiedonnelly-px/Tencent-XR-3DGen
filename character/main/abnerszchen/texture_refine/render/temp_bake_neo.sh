### bake pbr test
oid_dir="/apdcephfs_cq8/share_2909871/Assets/objaverse/mmd/render/mmd_20240518/part1_60k_0521/render_data/pod_1/objaverse/643e3c77db9c40c5a90f5cb78e3cd6c3/render_512_Valour"
obj_path="/aigc_cfs_6/Asset/objaverse/mesh/part1_360k/mesh/643e3c77db9c40c5a90f5cb78e3cd6c3/manifold/manifold.obj"
in_pose_json="${oid_dir}/cam_parameters.json"
transformation_txt="/aigc_cfs_2/Asset/objaverse/render/wonder3d_20240417/proc_data/pod_0/objaverse/643e3c77db9c40c5a90f5cb78e3cd6c3/proc_data/transformation.txt"
out_dir="/aigc_cfs_2/WSB/Data/debug/test_xcube/bake_senzhou_pbr/643e3c77db9c40c5a90f5cb78e3cd6c3"
mkdir -p ${out_dir}

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python temp_bake_neo_views.py ${obj_path} ${oid_dir}/color ${in_pose_json} ${out_dir}/bake_color --transformation_txt ${transformation_txt} --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt 0
python temp_bake_neo_views.py ${obj_path} ${oid_dir}/equilibrium/color ${in_pose_json} ${out_dir}/bake_equilibrium --transformation_txt ${transformation_txt} --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt 0
python temp_bake_neo_views.py ${obj_path} ${oid_dir}/roughness/color ${in_pose_json} ${out_dir}/bake_roughness --transformation_txt ${transformation_txt} --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt 0
python temp_bake_neo_views.py ${obj_path} ${oid_dir}/metallic/color ${in_pose_json} ${out_dir}/bake_metallic --transformation_txt ${transformation_txt} --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt 0


# ### bake pbr test
# oid_dir="/apdcephfs_cq8/share_2909871/Assets/objaverse/mmd/render/mmd_20240518/part1_60k/render_data/pod_0/objaverse/1a91607dc9e84eb591e47400c30a7229/render_512_Valour"
# obj_path="/aigc_cfs_6/Asset/objaverse/mesh/part1_360k/mesh/1a91607dc9e84eb591e47400c30a7229/manifold/manifold.obj"
# in_pose_json="${oid_dir}/cam_parameters.json"
# out_dir="/aigc_cfs_2/WSB/Data/debug/test_xcube/bake_senzhou_pbr/"
# mkdir -p ${out_dir}

# codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# cd ${codedir}

# python temp_bake_neo_views.py ${obj_path} ${oid_dir}/equilibrium/color ${in_pose_json} ${out_dir}/bake_equilibrium --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt 0 --keep_raw
# # python temp_bake_neo_views.py ${obj_path} ${oid_dir}/roughness/color ${in_pose_json} ${out_dir}/bake_roughness --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt 0
# # python temp_bake_neo_views.py ${obj_path} ${oid_dir}/metallic/color ${in_pose_json} ${out_dir}/bake_metallic --decimate_target -1 --tex_res 1024 --lrm_mode --temp_listcnt 0





# ### batch bake views test
# in_json="/aigc_cfs_2/WSB/Data/debug/test_xcube/new_neo_data.json"
# out_dir="/aigc_cfs_2/WSB/Data/debug/test_xcube/bake_senzhou/"
# mkdir -p ${out_dir}

# python temp_batch_bake_neo.py ${in_json} ${out_dir}/new_list1 --temp_listcnt 1 2>&1 | tee ${out_dir}/log1.txt
# # python temp_batch_bake_neo.py ${in_json} ${out_dir}/new_list2 --temp_listcnt 2 2>&1 | tee ${out_dir}/log2.txt
# # python temp_batch_bake_neo.py ${in_json} ${out_dir}/new_list3 --temp_listcnt 3 2>&1 | tee ${out_dir}/log3.txt

# # python temp_batch_bake_neo.py ${in_json} ${out_dir}/new_list4 --temp_listcnt 4 2>&1 | tee ${out_dir}/log4.txt


