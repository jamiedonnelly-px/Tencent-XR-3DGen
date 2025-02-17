decimate_target=10000
# decimate_target=-1

# ## lowpoly
# obj_path="/mnt/aigc_bucket_1/Asset/artcenter/lowpoly/mesh2/low_poly/Bear_A_Army/Bear_A_Army_manifold_full.obj"
# imgdir="/aigc_cfs/Asset/artcenter/low_poly_srender/render_data/Bear_A_Army/Bear_A_Army_Bear_A_Army_manifold_full_output_512_MightyWSB"
# outdir=/aigc_cfs_2/sz/proj/tex_cq/render/debug/shell_lowpoly
# extra_args=" --decimate_target ${decimate_target}"

# # obja 2048
# iod="28f8efb0c9c1445e860340cbfadf89a5"
# obj_path="/apdcephfs_cq8/share_2909871/Assets/objaverse/render_free/models/axisaligned/common_230k/models/${iod}/${iod}_manifold_full.obj"
# imgdir="/aigc_cfs/Asset/objaverse/render_free/weapons/hires_s/render_data/${iod}/${iod}_manifold_full_output_2048_MightyWSB"
# outdir=/aigc_cfs_2/sz/proj/tex_cq/render/debug/shell_obja_2048/${iod}
# extra_args="--lrm_mode --decimate_target ${decimate_target} --tex_res 2048"

iod="e0b44891919a458dbe44b9b661864877"
obj_path="/apdcephfs_cq8/share_2909871/Assets/objaverse/render_free/models/axisaligned/common_230k/models/${iod}/${iod}_manifold_full.obj"
imgdir="/aigc_cfs_4/Asset/objaverse/mass_production/part1_360k/render_data/0/${iod}/${iod}_manifold_full_output_512_MightyWSB"
outdir=/aigc_cfs_2/sz/proj/tex_cq/render/debug/texture_bake/${iod}
extra_args="--lrm_mode --decimate_target ${decimate_target} --tex_res 512"

# # obja
# obj_path="/apdcephfs_cq8/share_2909871/Assets/objaverse/render_free/models/axisaligned/common_230k/models/788f7de1e65a40d88e103a2424847e3c/788f7de1e65a40d88e103a2424847e3c_manifold_full.obj"
# imgdir="/aigc_cfs_4/Asset/objaverse/mass_production/part1_360k/render_data/26/788f7de1e65a40d88e103a2424847e3c/788f7de1e65a40d88e103a2424847e3c_manifold_full_output_512_MightyWSB"
# outdir=/aigc_cfs_2/sz/proj/tex_cq/render/debug/shell_obja_anti
# extra_args="--lrm_mode --decimate_target ${decimate_target} --tex_res 128"

# obj_path="/mnt/aigc_bucket_1/Asset/artcenter/lowpoly/mesh2/low_poly/Bear_A_Hazmat_B/Bear_A_Hazmat_B_manifold_full.obj"
# imgdir="/aigc_cfs/Asset/artcenter/low_poly_srender/render_data/Bear_A_Hazmat_B/Bear_A_Hazmat_B_Bear_A_Hazmat_B_manifold_full_output_512_MightyWSB"
# outdir=/aigc_cfs_2/sz/proj/tex_cq/render/debug/shell
# extra_args=""


codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}


python bake_to_one_tex.py ${obj_path} \
${imgdir}/emission/color \
${imgdir}/cam_parameters.json \
${outdir} ${extra_args}
