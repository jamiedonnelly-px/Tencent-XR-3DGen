

data_type="human"
if [ ${data_type} = "human" ]; then
    ### human 5.2w
    model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose9_argum/g4_text/design_only_all_b16a2_nsddpm"
    # model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/first_2k_b16a2_nsddpm"
    # in_json="/aigc_cfs_3/sz/data/tex/human/all_1222/creator/tex_creator_test.json"
    in_json="/aigc_cfs_3/sz/data/tex/human/design_lowpoly_vroid/design_only_withtext/tex_creator_test.json"
elif [ ${data_type}  = "objaverse" ]; then
    ### objaverse
    model_path="/aigc_cfs_3/sz/result/tex_creator/obja_gtD_srender_argum/g8/first_2k_b16a2_nsddpm"
    in_json="/aigc_cfs/sz/data/tex/weapon_srender/obj/tex_creator_test.json"
else
    echo "Invalid data_type: $data_type"
    exit 1
fi


out_dir=${model_path}/infer_in

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1

codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

python infer_creator.py ${model_path} ${in_json} ${out_dir} --condi_mode text
