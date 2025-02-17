import logging
import time
import json
import argparse
import os
import threading
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import sys
codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)
from main_call_texgen import TexgenInterface, init_job_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_blocking_call_query_text(args):
    texgen_interface, job_id, in_mesh_path, in_prompts, in_mesh_key, out_objs_dir, timeout = args
    fast_mode_param = {
        "fast_mode": True,
        "raw_glb": "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/mesh.glb",
        "out_glb": f"/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/replace_mesh_{job_id}.glb"
    }
    return texgen_interface.blocking_call_query_text(job_id,
                                                     in_mesh_path,
                                                     in_prompts,
                                                     in_mesh_key=in_mesh_key,
                                                     out_objs_dir=out_objs_dir,
                                                     fast_mode_param=fast_mode_param,
                                                     timeout=timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tqmd pulsar main consumer')
    parser.add_argument('--client_cfg_json',
                        type=str,
                        default='client_texgen.json',
                        help='relative name in codedir/configs')
    parser.add_argument('--model_name',
                        type=str,
                        default='uv_mcwy',
                        help='select model. can be uv_mcwy, control_mcwy, imguv_mcwy, imguv_lowpoly, pipe_type_dataset')
    args = parser.parse_args()
    ### prepare example data
    # in_mesh_path = f"/aigc_cfs_gdp/sz/batch_1012/test/newglb/textured_1/skirt_4_panels_BSPC1JNS6U_skirt_4_panels_BSPC1JNS6U.glb"
    in_mesh_path = f"/aigc_cfs_gdp/Asset/designcenter/clothes/convert/mcwy2/remove_skin_mesh/meshes/Top/BR_TOP_1_F_T/BR_TOP_1_fbx2020.obj"
    in_prompts = "Clothing with white stripes and red."
    in_condi_img = "/aigc_cfs_gdp/Asset/designcenter/clothes/convert/mcwy2/remove_skin_mesh/meshes/Top/BR_TOP_1_F_T/BR_TOP_1_Albedo.png"
    in_mesh_key = "DAZ_Paladin_and_Paragon_2022_Complete__bottom_Bottom"
    out_name_temp = in_prompts.replace(" ", "_")
    out_objs_dir = f"/aigc_cfs_gdp/sz/batch_1012/test/fastmode/out_daz/fast_{in_prompts}"

    # 1. init
    texgen_interface = TexgenInterface(args.client_cfg_json, args.model_name)

    # # senbo mode
    # extra_param = {"preprocess": "add_image"}
    # job_id = init_job_id()
    # success_flag, result_meshs = texgen_interface.blocking_call_query_text(
    #     job_id,
    #     "/aigc_cfs_gdp/WSB/sewfactory/test/300_temp/sewfactory/dress_sleeveless_005CWF73E9/result/preprocess.obj",
    #     in_prompts,
    #     in_mesh_key=None,
    #     out_objs_dir="/aigc_cfs_gdp/sz/batch_1107/cloth/out_direct1",
    # )
    # # success_flag, result_meshs = texgen_interface.blocking_call_query_text(
    # #     job_id,
    # #     "/aigc_cfs_gdp/WSB/sewfactory/dress_sleeveless_00F3PNYG6L/static/dress_sleeveless_00F3PNYG6L_dress_sleeveless_00F3PNYG6L.obj",
    # #     in_prompts,
    # #     in_mesh_key=None,
    # #     out_objs_dir="/aigc_cfs_gdp/sz/batch_1107/cloth/out_direct",
    # #     extra_param=extra_param,
    # # )
    # assert success_flag
    # print('result_meshs ', result_meshs)

    job_id = init_job_id()
    mid_result_dir = os.path.join("/aigc_cfs_gdp/jiawei/data/texture_generation/", job_id)
    fast_mode_param = {
        "fast_mode": True,
        "raw_glb": "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/mesh.glb",
        "out_glb": f"/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/replace_mesh_{job_id}.glb"
    }
    input_param = [
        {
            "in_mesh_key": "VRoid_4_4423006740960699034_Top",
            "in_mesh_path": None,
            "in_prompts": "red",
            "in_condi_img": None,
            "mode": "text"
        },
        {
            "in_mesh_key": "BTM_419",
            "in_mesh_path": None,
            "in_prompts": "",
            "in_condi_img": "/aigc_cfs_gdp/sz/batch_1106/in_imgs/0a1de9108cabcf6eea187536df3c627f.jpg",
            "mode": "image"
        },
        {
            "in_mesh_key": "SH_160",
            "in_mesh_path": None,
            "in_prompts": "red",
            "in_condi_img": "/aigc_cfs_gdp/sz/batch_1106/in_imgs/3c83e2b7599cc89a73a620662c878100.jpg",
            "mode": "mix"
        },
    ]
    # batch fast mode
    success_flag, result_meshs = texgen_interface.blocking_call_batch_query(
        job_id,
        input_param,    # list of dict
        mid_result_dir=mid_result_dir,   # 存中间结果的目录，比如 根目录+job_id
        fast_mode_param=fast_mode_param,
    )
    assert success_flag
    print('blocking_call_batch_query result_meshs ', result_meshs)
    # breakpoint()




    # # once fast mode
    # meta = input_param[0]
    # success_flag, result_pngs = texgen_interface.blocking_call_infer_fast_uv_once(job_id,
    #                               in_mesh_path= meta["in_mesh_path"],
    #                               in_mesh_key= meta["in_mesh_key"],
    #                               out_objs_dir= os.path.join(mid_result_dir, "call_0"),
    #                               in_prompts= meta["in_prompts"],
    #                               in_condi_img= meta["in_condi_img"],
    #                               )
    # assert success_flag
    

    # 2. 堵塞式调用, 类似grpc
    use_t_dict = {}
    ts = time.time()
    job_id = init_job_id()
    fast_mode_param = {
        "fast_mode": True,
        "raw_glb": "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/mesh.glb",
        "out_glb": f"/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/replace_mesh_{job_id}.glb"
    }
    # fast mode
    success_flag, result_meshs = texgen_interface.blocking_call_query_text(
        job_id,
        in_mesh_path,
        in_prompts,
        in_mesh_key=in_mesh_key,
        out_objs_dir=os.path.join(out_objs_dir, f"text_fast_mode"),
        fast_mode_param=fast_mode_param,
    )
    print(f'use_t ', time.time() - ts)
    assert success_flag

    # original mode
    success_flag, result_meshs = texgen_interface.blocking_call_query_text(
        job_id,
        in_mesh_path,
        in_prompts,
        in_mesh_key=in_mesh_key,
        out_objs_dir=os.path.join(out_objs_dir, f"text_fast_mode"),
    )
    assert success_flag

    job_id = init_job_id()
    fast_mode_param = {
        "fast_mode": True,
        "raw_glb": "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/mesh.glb",
        "out_glb": f"/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/replace_mesh_{job_id}.glb"
    }
    success_flag, result_meshs = texgen_interface.blocking_call_query_image(job_id,
                                                            in_mesh_path,
                                                            in_condi_img,
                                                            in_mesh_key=in_mesh_key,
                                                            out_objs_dir=os.path.join(out_objs_dir, "image"),
                                                            fast_mode_param=fast_mode_param,)
    assert success_flag
    job_id = init_job_id()
    fast_mode_param = {
        "fast_mode": True,
        "raw_glb": "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/mesh.glb",
        "out_glb": f"/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/23e04bc2-feee-57f4-b525-3684e08a95a9/mesh/replace_mesh_{job_id}.glb"
    }
    success_flag, result_meshs = texgen_interface.blocking_call_query_text_image(job_id,
                                                            in_mesh_path,
                                                            in_prompts,
                                                            in_condi_img,
                                                            in_mesh_key=in_mesh_key,
                                                            out_objs_dir=os.path.join(out_objs_dir, "mix"))
    assert success_flag
    logging.info("test once done")

    # 3. multi-threading
    test_multi_thread = True
    if test_multi_thread:
        num_threads = 4
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            args_list = [(texgen_interface, init_job_id(), in_mesh_path, in_prompts, in_mesh_key, out_objs_dir + f"_{i}",
                        300) for i in range(num_threads)]

            futures = [executor.submit(run_blocking_call_query_text, args) for args in args_list]

            final_results = [future.result() for future in futures]

        for result in final_results:
            print(result)

    logging.info("test multi-process done")

    texgen_interface.close()

    logging.info("test done")
