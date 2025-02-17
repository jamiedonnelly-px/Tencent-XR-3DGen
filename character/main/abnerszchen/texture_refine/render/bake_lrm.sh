# decimate_target=10000
decimate_target=-1

root_dir="/aigc_cfs/weizhe/code/git_moa/InstantMesh/outputs/instant-mesh-large/meshes"
obj_path="${root_dir}/hatsune_miku.obj"
outdir="${root_dir}/bake"
extra_args=" --decimate_target ${decimate_target} --tex_res 1024 --keep_raw"


codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}


python test_bake_lrm.py ${obj_path} \
${outdir} ${extra_args}
