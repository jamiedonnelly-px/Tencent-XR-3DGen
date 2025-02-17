import time
import os
import argparse
import logging
import grpc
import teximguv_pb2
import teximguv_pb2_grpc
from easydict import EasyDict as edict

import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from dataset.utils_dataset import load_json

class TeximguvClient():
    def __init__(self, client_cfg_json='client_teximguv.json', model_name='mcwy', device='cuda'):
        """client

        Args:
            client_cfg_json: relative name in codedir/configs. Defaults to 'client_teximguv.json'.
            model_name: key of cfg, can be selected in webui
            device
        """
        self.load_run_cfg(client_cfg_json, model_name=model_name)
        self.device = device
        self.run_cfg.server_addr

    def load_run_cfg(self, cfg_json, model_name=None):
        """set self.model_name_list and self.run_cfg from json

        Args:
            cfg_json: relative name in codedir/configs
            model_name: model key. Defaults to None.
        """
        logging.info(f"input of load_run_cfg: cfg_json: {cfg_json}, model_name:{model_name}")
        self.configs_dir = os.path.join(codedir, 'configs')
        cfg_json_ = os.path.join(self.configs_dir, cfg_json)
        if not os.path.exists(cfg_json_):
            cfg_json_ = os.path.join(self.configs_dir, "client_teximguv.json")
        assert os.path.exists(cfg_json_), f"can not find valid cfg_json: {cfg_json}, {cfg_json_}"
        cfg_root = load_json(cfg_json_) #  
        assert len(cfg_root), f"invalid cfg_json: {cfg_json}"

        model_name_list= list(cfg_root.keys())
        if model_name is not None and model_name in cfg_root:
            run_cfg = cfg_root[model_name]
        else:
            logging.warn(f"invalid model_name in cfg_json: {cfg_json}, model_name:{model_name}, juet select first")
            model_name = model_name_list[0]
            run_cfg = cfg_root[model_name]

        self.model_name_list = model_name_list
        self.run_cfg = edict(run_cfg)
        self.cfg_json = cfg_json_
        self.model_name = model_name
        logging.info(f"load_run_cfg done with self.model_name:{self.model_name}")
        return

    def query_job(self, stub, job_id, max_cnt=100):
        """Query the status of the job until it is finished or failed

        Args:
            stub (PandoraxStub): stub of the channel
            job_id (int): unique id for the job
        Return:
            results(string), can be out_obj
        """
        responses = stub.QueryJobs(
            teximguv_pb2.JobStatusRequest(job_ids=[job_id]
                                            ))

        cnt = 0
        while responses.job_status[0].status not in [teximguv_pb2.FINISHED, teximguv_pb2.FAILED] and cnt < max_cnt:
            print(responses.job_status[0].status, end='')
            time.sleep(1)

            responses = stub.QueryJobs(
                teximguv_pb2.JobStatusRequest(job_ids=[job_id]
                                                ))
            cnt += 1

        print('query done')
        if not responses or len(responses.job_status) < 1:
            print('ERROR in valid responses')
            return ''

        return responses.job_status[0].results

    
    def warp_output(self, out_objs_queryed):
        """_summary_

        Args:
            out_objs_queryed: If '' is returned, the task failed. If run success, it is ",".join(out_obj_paths)
        Return list of obj path
        """
        if not out_objs_queryed:
            return ['']
        out_obj_paths = out_objs_queryed.split(',')
        
        return out_obj_paths
    
    def test_add(self):
        logging.info(f"run test_add at: {self.run_cfg.server_addr}")
        with grpc.insecure_channel(self.run_cfg.server_addr) as channel:
            stub = teximguv_pb2_grpc.TeximguvStub(channel)
            response = stub.Add(teximguv_pb2.AdditionRequest(x=5, y=3))
            print("5 + 3 =", response.result)

    def client_obj_tex_imguv(self, in_obj, out_objs_dir, in_prompts='', in_condi_img='',
                          uv_res=512, num_inference_steps=20, guidance_scale=7.5, debug_save=True):
        """query grpc server of tex imguv

        Args:
            in_obj(string): raw obj path with uv coord
            out_objs_dir(string): output dir with multi output objs
            in_prompts(string): input text, list of text (merged with magic key :::) or ''
            in_condi_img(string): condi img path or ''

        Returns:
            out_obj_paths: list of out_obj_path
            # out_objs_queryed(string): If '' is returned, the task failed. If run success, it is ",".join(out_obj_paths)
        """
        query_lj_ip_port = self.run_cfg.server_addr
        with grpc.insecure_channel(query_lj_ip_port) as channel:
            stub = teximguv_pb2_grpc.TeximguvStub(channel)
            logging.info(f"begin client_obj_tex_imguv in ip {query_lj_ip_port}")
           
            response = stub.NewJob(
                teximguv_pb2.JobRequest(
                    task_type=teximguv_pb2.TEX_IMGUV,
                    in_obj=in_obj,
                    out_objs_dir=out_objs_dir,
                    in_prompts=in_prompts,
                    in_condi_img=in_condi_img,
                    uv_res=uv_res,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    debug_save=debug_save,
                )
            )
            job_id = response.job_id

            out_objs_queryed = self.query_job(stub, job_id)
            print('debug, get new out_objs_queryed:', out_objs_queryed)
            if not out_objs_queryed:
                print('Failed! ', out_objs_queryed)
                return ['']
                
            out_obj_paths = self.warp_output(out_objs_queryed)
        return out_obj_paths


def test_client(client_cfg_json, model_key):
    client = TeximguvClient(client_cfg_json, model_key)
    print(f'lj_ip_port: {client.run_cfg.server_addr}')
    client.test_add()

    uv_res=512
    num_inference_steps=20
    guidance_scale=7.5
    debug_save=True
    if model_key == 'mcwy':
        # debug objaverse weapon
        in_obj = f"/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/MCWY_2_Dress/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/uv_condition/mesh.obj"
        out_objs_dir = "/aigc_cfs_3/sz/server/tex_imguv/client_log/mcwy_debug"
        in_prompts = ''
        in_condi_img = "/aigc_cfs/Asset/designcenter/clothes/render_part2/render_data/dress/render_data/DR_640_F_A/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/color/cam-0032.png"
        # out_debug_dir=os.path.dirname(out_obj)
    elif model_key == 'lowpoly':
        # oname = '012c38ecb7f9308e805decd077adbef6c9d31af8_manifold_full_output_512_MightyWSB'
        # in_condi = f'/aigc_cfs_3/sz/data/tex/human/all_1222/Designcenter_1/{oname}/color/cam-0100.png'
        in_obj = "/aigc_cfs/sz/data/tex/lowpoly/low_poly/Dog_C_Police/uv_condition/mesh.obj"
        out_objs_dir = "/aigc_cfs_3/sz/server/tex_imguv/client_log/lowploy_debug"
        # in_prompts = ''
        in_prompts = ":::".join(["", "low poly"])
        in_condi_img = "/aigc_cfs/Asset/artcenter/low_poly_srender/render_data/Dog_C_Police/Dog_C_Police_Dog_C_Police_manifold_full_output_512_MightyWSB/color/cam-0037.png"        
        # out_obj = '/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm/new_objs_test_1_pipe/Designcenter_1/3k/obj_3k/xatlas/out/mesh.obj'

    else:
        print('invalid model_key')
        exit()

    out_obj_paths = client.client_obj_tex_imguv(
        in_obj=in_obj,
        out_objs_dir=out_objs_dir,
        in_prompts=in_prompts,
        in_condi_img=in_condi_img,
        uv_res=uv_res,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        debug_save=debug_save,
    )
    print("out_obj_paths ", out_obj_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('--client_cfg_json', type=str, default='client_teximguv.json', help='relative name in codedir/configs')
    parser.add_argument('--model_key', type=str, default='mcwy', help='select model. can be mcwy, lowpoly')
    args = parser.parse_args()

    test_client(args.client_cfg_json, args.model_key)
