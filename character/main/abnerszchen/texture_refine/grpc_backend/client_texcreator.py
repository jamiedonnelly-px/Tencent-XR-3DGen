import time
import os
import argparse
import grpc
import texcreator_pb2
import texcreator_pb2_grpc

lj_ip = "localhost" 
lj_port = "8080"
lj_ip_port = f"{lj_ip}:{lj_port}"


def query_job(stub, job_id, max_cnt=100):
    """Query the status of the job until it is finished or failed

    Args:
        stub (PandoraxStub): stub of the channel
        job_id (int): unique id for the job
    Return:
        results(string), can be out_obj
    """
    responses = stub.QueryJobs(
        texcreator_pb2.JobStatusRequest(job_ids=[job_id]
                                        ))

    cnt = 0
    while responses.job_status[0].status not in [texcreator_pb2.FINISHED, texcreator_pb2.FAILED] and cnt < max_cnt:
        print(responses.job_status[0].status, end='')
        time.sleep(1)

        responses = stub.QueryJobs(
            texcreator_pb2.JobStatusRequest(job_ids=[job_id]
                                            ))
        cnt += 1

    print('query done')
    if not responses or len(responses.job_status) < 1:
        print('ERROR in valid responses')
        return ''

    return responses.job_status[0].results


def test_add():
    print('lj_ip_port ', lj_ip_port)
    with grpc.insecure_channel(lj_ip_port) as channel:
        stub = texcreator_pb2_grpc.TexcreatorStub(channel)
        response = stub.Add(texcreator_pb2.AdditionRequest(x=5, y=3))
        print("5 + 3 =", response.result)


def run_obj_tex_creator(query_lj_ip_port, in_obj, in_condi, out_obj, out_debug_dir='', debug_paste_condi='false'):
    """query grpc server of tex creator

    Args:
        query_lj_ip_port: ip addr, {lj_ip}:{lj_port}
        in_obj: obj path
        in_condi: condition image path
        out_obj: out obj path
        out_debug_dir: if not empty, save debug result(gif and vis image). Defaults to ''.

    Returns:
        out_obj_queryed: If '' is returned, the task failed. If a valid address is returned, it is out_obj path
    """
    with grpc.insecure_channel(query_lj_ip_port) as channel:
        stub = texcreator_pb2_grpc.TexcreatorStub(channel)
        print(f"begin run_obj_tex_creator in ip {query_lj_ip_port}")

        response = stub.NewJob(
            texcreator_pb2.JobRequest(
                task_type=texcreator_pb2.TEX_CREATOR,
                in_obj=in_obj, in_condi=in_condi, out_obj=out_obj, out_debug_dir=out_debug_dir, debug_paste_condi=debug_paste_condi,
            ))
        job_id = response.job_id

        out_obj_queryed = query_job(stub, job_id)
        print('debug, get new out_obj_queryed ', out_obj_queryed)
        if not out_obj_queryed:
            print('Failed! ', out_obj_queryed)
    return out_obj_queryed


def test_client(model_key):
    print(f'lj_ip_port: {lj_ip_port}')
    test_add()

    # model_key = 'tex_creator_weapon'
    # model_key = 'tex_creator_human_design'

    if model_key == 'tex_creator_weapon':
        # debug objaverse weapon
        oname = '6454fb033a6b486085fbd7496749be11_manifold_full_output_512_MightyWSB'
        in_obj = f'/aigc_cfs/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdfcolor_objaverse_kl_v0.0.0/triplane_2023-12-21-20:39:16/objaverse/{oname}/0000/mesh.obj'
        in_condi = f'/aigc_cfs/Asset/objaverse/render_free/weapons/lrm/render_data/6454fb033a6b486085fbd7496749be11/{oname}/color/cam-0024.png'
        out_obj = f'/aigc_cfs_3/sz/result/tex_creator/obja_gtD_srender_argum/g8/first_2k_b16a2_nsddpm/new_objs_test_1_pipe/objaverse/{oname}/mesh.obj'
        # out_debug_dir=os.path.dirname(out_obj)
        out_debug_dir = ''
        debug_paste_condi = 'false'
    elif model_key == 'tex_creator_human_design9':
        in_dir = "/aigc_cfs_2/neoshang/data/web_character_tmp_v2/20240403/1605531eb411f4f46844e2987c79b513417ffb/0"
        # in_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_images_tmp1_mesh/20240126/1431187e6b8ebe207d43fdacc5325871bc18dc/0"
        in_condi = os.path.join(in_dir, '0.jpg')
        # in_condi = os.path.join(in_dir, 'condi.png')
        in_obj = os.path.join(in_dir, 'mesh.obj')
        out_obj = os.path.join(in_dir, 'creator_posknn9/mesh.obj')
        out_debug_dir = ''
        debug_paste_condi = 'false' # 'true' or 'false
        
    elif model_key == 'tex_creator_human_design':
        # oname = '012c38ecb7f9308e805decd077adbef6c9d31af8_manifold_full_output_512_MightyWSB'
        # in_condi = f'/aigc_cfs_3/sz/data/tex/human/all_1222/Designcenter_1/{oname}/color/cam-0100.png'
        # in_obj = '/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm/new_objs_test_1_pipe/Designcenter_1/3k/obj_3k/xatlas/mesh.obj'
        # out_obj = '/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm/new_objs_test_1_pipe/Designcenter_1/3k/obj_3k/xatlas/out/mesh.obj'
        
        # in_dir = "/aigc_cfs_2/neoshang/data/web_character_tmp_v2/20240403/1605531eb411f4f46844e2987c79b513417ffb/0"
        in_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_images_tmp1_mesh/20240126/1431187e6b8ebe207d43fdacc5325871bc18dc/0"
        # in_dir = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/test_images_tmp1_mesh/20240126/143127a5a0500155ab425fa03f64dbacb57778/0"
        in_condi = os.path.join(in_dir, 'condi.png')
        in_obj = os.path.join(in_dir, 'mesh.obj')
        out_obj = os.path.join(in_dir, 'creator_uvknn8/mesh.obj')
        # in_obj = f'/aigc_cfs/weixuan/code/DiffusionSDF/config/stage1_vae_geotri_people_alldata/recon2024-01-02-14:57:12/Designcenter_1/{oname}/obj/mesh.obj'
        # out_obj = f'/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm/new_objs_test_1_pipe/Designcenter_1/{oname}/mesh.obj'
        out_debug_dir = ''
        debug_paste_condi = 'false' # 'true' or 'false
        # out_debug_dir=os.path.dirname(out_obj)
    elif model_key == 'tex_creator_human_lowpoly':
        in_condi = "/aigc_cfs/weixuan/code/DiffusionSDF/config/stage1_vae_geotri_people_finetune_20240119/recon2024-01-22-15:39:18/low_poly/Bear_A_Army_Bear_A_Army_manifold_full_output_512_MightyWSB/cam-0050.png"
        in_obj = "/aigc_cfs/weixuan/code/DiffusionSDF/config/stage1_vae_geotri_people_finetune_20240119/recon2024-01-22-15:39:18/low_poly/Bear_A_Army_Bear_A_Army_manifold_full_output_512_MightyWSB/obj/mesh.obj"
        out_obj = os.path.join("/aigc_cfs_3/sz/result/tex_creator/client_debug/lowpoly/", 'Bear/mesh.obj')
        # in_obj = f'/aigc_cfs/weixuan/code/DiffusionSDF/config/stage1_vae_geotri_people_alldata/recon2024-01-02-14:57:12/Designcenter_1/{oname}/obj/mesh.obj'
        # out_obj = f'/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm/new_objs_test_1_pipe/Designcenter_1/{oname}/mesh.obj'
        out_debug_dir = ''
        debug_paste_condi = 'false' # 'true' or 'false
        # out_debug_dir=os.path.dirname(out_obj)
    else:
        print('invalid model_key')
        exit()

    out_obj_queryed = run_obj_tex_creator(lj_ip_port, in_obj, in_condi, out_obj, out_debug_dir, debug_paste_condi=debug_paste_condi)
    print('out_obj_queryed ', out_obj_queryed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('--model_key', type=str, default='tex_creator_human_design9',
                        help='select model. can be tex_creator_human_design9, tex_creator_weapon, tex_creator_human_design or other keys in cfg json')
    args = parser.parse_args()

    test_client(args.model_key)
