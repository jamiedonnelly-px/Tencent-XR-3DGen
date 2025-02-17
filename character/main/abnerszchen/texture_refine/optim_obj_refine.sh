
in_model_path="/aigc_cfs_2/sz/result/tex/condi_g8/first_300_b8a2"
in_obj="/apdcephfs_cq8/share_2909871/shenzhou/result/gen/b1/sz_diffusion_4096_v0_test/first_2k/guofenggame/C0019_clean_gaojishiwei/0000/mesh.obj"
in_condi="/apdcephfs_cq8/share_2909871/shenzhou/result/gen/b1/sz_diffusion_4096_v0_test/first_2k/guofenggame/C0019_clean_gaojishiwei/cam-0100.png"
out_dir=${in_model_path}/new_objs

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python run_optim_texrefine.py ${in_model_path} ${in_obj} ${in_condi} ${out_dir}