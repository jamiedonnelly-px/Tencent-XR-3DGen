import time
import os
import argparse
import logging
import grpc

from easydict import EasyDict as edict

import sys
codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

import grpc_backend.srgen_pb2 as srgen_pb2
import grpc_backend.srgen_pb2_grpc as srgen_pb2_grpc

#from dataset.utils_dataset import load_json

class SrGenClient():
    def __init__(self, server_addr="ip_addr:80", device='cuda'):
        """client

        Args:
            client_cfg_json: abs path or relative name in codedir/configs. Defaults to 'client_texgen.json'.
            model_name: key of cfg, can be selected in webui
            device
        """
        #self.load_run_cfg(client_cfg_json, model_name=model_name)
        self.device = device
        self.server_addr = server_addr
        self.log_root_dir = "/aigc_cfs/xibinsong/code/imgsr/logs"

    def reset(self):
        # TODO
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
            srgen_pb2.JobStatusRequest(job_ids=[job_id]
                                            ))

        cnt = 0
        while responses.job_status[0].status not in [srgen_pb2.FINISHED, srgen_pb2.FAILED] and cnt < max_cnt:
            print(responses.job_status[0].status, end='')
            time.sleep(1)

            responses = stub.QueryJobs(
                srgen_pb2.JobStatusRequest(job_ids=[job_id]
                                                ))
            cnt += 1

        print('query done')
        if not responses or len(responses.job_status) < 1:
            print('ERROR in valid responses')
            return ''

        return responses.job_status[0].results

    def wrap_output(self, out_mesh_paths_queryed):
        """_summary_

        Args:
            out_mesh_paths_queryed: If '' is returned, the task failed. If run success, it is ",".join(out_mesh_paths)
        Return list of obj path
        """
        if not out_mesh_paths_queryed:
            return ['']
        out_mesh_paths = out_mesh_paths_queryed.split(',')

        return out_mesh_paths

    def test_add(self):
        logging.info(f"run test_add at: {self.server_addr}")
        with grpc.insecure_channel(self.server_addr) as channel:
            stub = srgen_pb2_grpc.SrgenStub(channel)
            response = stub.Add(srgen_pb2.AdditionRequest(x=5, y=3))
            print("srgen 5 + 3 =", response.result)

    # core function of grpc
    def client_sr(
        self,
        input_path,
        out_img_dir,
        up_scale,
    ):
        """query grpc server of tex imguv

        Args:
            pipe_type(string): need be tex_control or tex_imguv, select pipeline
            in_mesh_path(string): raw obj/glb path with uv coord
            out_objs_dir(string): output dir with multi output objs
            in_prompts(string): input text, list of text (merged with magic key :::) or ''
            in_condi_img(string): condi img path or ''
            in_obj_type(string): obj_type or '' TODO(csz, not used now)

        Returns:
            out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed. 
        """
        query_lj_ip_port = self.server_addr
        with grpc.insecure_channel(query_lj_ip_port) as channel:
            stub = srgen_pb2_grpc.SrgenStub(channel)

            task_type = srgen_pb2.IMG_SR

            logging.info(f"begin client_sr in ip {query_lj_ip_port}, with task_type {task_type}")

            #task_type=task_type,
            #print("333333333")
            #print(up_scale)

            # refer data to protos/texgen.proto message JobRequest; func to NewJob in server request.xxx
            response = stub.NewJob(
                srgen_pb2.JobRequest(
                    task_type=task_type,
                    input_path=input_path,
                    out_img_dir=out_img_dir,
                    up_scale=up_scale,
                )
            )
            job_id = response.job_id

            out_mesh_paths_queryed = self.query_job(stub, job_id)
            print('debug, get new out_mesh_paths_queryed:', out_mesh_paths_queryed)
            if not out_mesh_paths_queryed:
                print('Failed! ', out_mesh_paths_queryed)
                return ['']

            out_mesh_paths = self.wrap_output(out_mesh_paths_queryed)
        return out_mesh_paths

    # interface of web ui
    def webui_query_image(
        self,
        input_path,
        out_dir,
        up_scale,
    ):
        """webui query image only mode

        Args:
            in_mesh_path(string): raw obj/glb path with uv coord
            in_condi_img(string): condi img path
            out_objs_dir(string): output dir with multi output objs
            in_obj_type(string): obj_type or '' TODO(csz, not used now)
            pipe_type(string): need be tex_control or tex_imguv, select pipeline
        Returns:
            out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
        """

        out_mesh_paths = self.client_sr(
            input_path = in_img_path,
            out_img_dir = out_dir,
            up_scale = up_scale,
        )
        return out_mesh_paths


def test_client():
    client = SrGenClient()
    print(f'lj_ip_port: {client.server_addr}')
    client.test_add()

    in_img_path = "/aigc_cfs_3/sz/result/tex_creator/human/pose9_argum/g8/design_lowpoly_vroid_all_b16a2_nsddpm/new_objs_5_condi_force/Designcenter_1/02c9f5db681b34f75bf0c12b574187913523c4a7_manifold_full_output_512_MightyWSB/bake/cam-0002.png"
    # in_img_path = "/aigc_cfs/xibinsong/code/imgsr/data/test_gen_img/texture_kd.png"
    out_dir = "/aigc_cfs/xibinsong/code/imgsr/sz"
    os.makedirs(out_dir, exist_ok=True)
    up_scale = 2

    print("up_scale:" + str(up_scale))

    out_mesh_paths = client.client_sr(
        input_path = in_img_path,
        out_img_dir = out_dir,
        up_scale = up_scale,
    )
    print("out_mesh_paths ", out_mesh_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    #parser.add_argument('--client_cfg_json', type=str, default='client_texgen.json', help='relative name in codedir/configs')
    #parser.add_argument('--model_key', type=str, default='control_mcwy', help='select model. can be control_ready, control_mcwy, imguv_mcwy, imguv_lowpoly')
    args = parser.parse_args()

    test_client()
