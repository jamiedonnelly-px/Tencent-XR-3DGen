# decimate_target=3000
decimate_target=-1


iod="02d76c05d11af63e66cb6fa5cbfeee44afe03326_manifold_full_output_512_MightyWSB"
model_out="/aigc_cfs_3/sz/result/tex_creator/human/pose9_argum/g8/design_lowpoly_vroid_all_b16a2_nsddpm/new_objs_5_condi_force"
in_pose_json="/aigc_cfs_2/sz/proj/tex_cq/data/cams/cam_parameters_human9_reid.json"

# model_out="/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm/new_objs_5_condi_force"
# in_pose_json="/aigc_cfs_2/sz/proj/tex_cq/data/cams/cam_parameters_human8_reid.json"

mesh_root="/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdf_character_kl_v1.0.0/triplane_2024-01-15-15:33:22/Designcenter_1"
debug_dir="${model_out}/Designcenter_1/${iod}"
obj_path="${mesh_root}/${iod}/0000/mesh.obj"
imgdir="${debug_dir}/bake"
# outdir="/aigc_cfs_gdp/sz/result/pipe_test/82941b42-a516-4fbc-b3ed-9bcc94754bbf/use_raw/bake_normal"
outdir="${debug_dir}/new_bake_${decimate_target}/norm_cos0.25_neww2c"
extra_args="--decimate_target ${decimate_target} --tex_res 1024 --keep_raw"


codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

mkdir -p ${outdir}

python bake_to_one_tex.py ${obj_path} \
${imgdir} \
${in_pose_json} \
${outdir} ${extra_args} 2>&1 | tee ${outdir}/log.txt
