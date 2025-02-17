
codedir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${codedir}

data_type="human_test"
if [ ${data_type} = "human" ]; then
    ### human 5.2w
    # in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/first_2k_b16a2_nsddpm"
    # in_dataset_json="/aigc_cfs_3/sz/data/tex/human/all_1222/creator/vae/test.json"
    # in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm"
    # in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose9_argum/g8/design_lowpoly_vroid_all_b16a2_nsddpm"
    in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose9_argum/g4/design_only_all_b16a2_nsddpm"

    in_dataset_json="/aigc_cfs_3/sz/data/tex/human/all_1222/creator/diffusion/test.json"    
    # in_dataset_json="/aigc_cfs_3/sz/data/tex/human/all_1222/creator/vae/test.json"    
    # in_dataset_json="/aigc_cfs_3/sz/data/tex/human/all_1222/creator_design_only/tex_creator_test.json"    
    # extra_args="--pose_json ${codedir}/data/cams/cam_parameters_human8.json"
    extra_args="--pose_json ${codedir}/data/cams/cam_parameters_human9.json"
elif [ ${data_type}  = "human_test" ]; then
    in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose9_argum/g4/design_only_all_b16a2_nsddpm"
    in_dataset_json="/aigc_cfs_2/neoshang/code/aigc_webui/web_character_tmp_v2/20240408_demo/test.json"
    extra_args="--pose_json ${codedir}/data/cams/cam_parameters_human9.json"  
elif [ ${data_type}  = "lowpoly" ]; then
    ### objaverse
    in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/low_poly/g8_5w_lpvd_b16a2_nsddpm"
    # in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/low_poly/g8_lponly__all_b16a2_nsddpm"
    in_dataset_json="/aigc_cfs_3/sz/data/tex/human/low_poly/test_infer_obj.json"
    extra_args="--pose_json ${codedir}/data/cams/cam_parameters_human8.json"  
elif [ ${data_type}  = "design_lowpoly_vroid" ]; then
    ### objaverse
    in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/pose9_argum/g8/design_lowpoly_vroid_all_b16a2_nsddpm"
    # in_model_path="/aigc_cfs_3/sz/result/tex_creator/human/low_poly/g8_lponly__all_b16a2_nsddpm"
    # in_dataset_json="/aigc_cfs_3/sz/data/tex/human/low_poly/test_infer_obj.json"
    in_dataset_json="/aigc_cfs_3/sz/data/tex/human/all_1222/creator/diffusion/test.json"
    extra_args="--pose_json ${codedir}/data/cams/cam_parameters_human9.json"        
elif [ ${data_type}  = "objaverse" ]; then
    ### objaverse
    in_model_path="/aigc_cfs_3/sz/result/tex_creator/obja_gtD_srender_argum/g8/first_2k_b16a2_nsddpm"
    in_dataset_json="/aigc_cfs/sz/data/tex/weapon_srender/obj/tex_creator_test.json"
    # in_dataset_json="/aigc_cfs/sz/data/tex/weapon_srender/obj/tex_creator_train.json"
    extra_args="--lrm_mode --pose_json ${codedir}/data/cams/cam_parameters_srender8.json"
else
    echo "Invalid data_type: $data_type"
    exit 1
fi

infer_cnt=5
out_dir=${in_model_path}/new_objs_${infer_cnt}_sr_bake

mkdir -p ${out_dir}
log_txt=${out_dir}/log.txt
exec > >(tee ${log_txt}) 2>&1


cmd="python run_optim_texcreator.py ${in_model_path} ${in_dataset_json}  ${out_dir}  ${extra_args} --infer_cnt ${infer_cnt}"
echo ${cmd}
eval ${cmd}
