
model_path="/aigc_cfs_2/sz/result/tex/condi_g8/first_300_b8a2"
in_json="/apdcephfs_cq8/share_2909871/shenzhou/data/tex_refine/debug/tex_refine_test.json"
out_dir=${model_path}/infer

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python infer.py ${model_path} ${in_json} ${out_dir}