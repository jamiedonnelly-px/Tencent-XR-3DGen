
model_path="/aigc_cfs/sz/result/tex/condi_g1/first_300_b8a1_ddpm"
in_json="/aigc_cfs/sz/data/tex/first_300/tex_refine_test.json"
out_dir=${model_path}/infer

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python infer.py ${model_path} ${in_json} ${out_dir}