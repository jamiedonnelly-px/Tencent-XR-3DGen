import logging
import argparse
import os
import sys
import grpc
import teximguv_pb2
import teximguv_pb2_grpc
from concurrent import futures
from easydict import EasyDict as edict
import time
import threading

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from dataset.utils_dataset import load_json
from pipe_teximguv import ObjTexImgUVPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class TeximguvServer(teximguv_pb2_grpc.TeximguvServicer):
    def __init__(self, cfg_json='teximguv.json', model_name='mcwy', device='cuda'):
        """server

        Args:
            cfg_json: relative name in codedir/configs. Defaults to 'teximguv.json'.
            model_name: key of cfg, can be selected in webui
        """
        super(TeximguvServer, self).__init__()
        self.load_cfg(cfg_json, model_name=model_name)
        self.device = device
        self.jobs = {}
        self.job_cnt = 0
        self.lock = threading.Lock()
        self.retry = 3

        self.log_dir = os.path.join(self.cfg_model.log_root_dir, time.strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(self.log_dir, exist_ok=True)

        logging.info(f'begin loading {self.model_name} model from {self.cfg_model.in_model_path}, Need a little time in T10')
        self.pipeline = ObjTexImgUVPipeline(self.cfg_model.in_model_path, self.cfg_model.in_sd_path, 
                                            self.cfg_model.ip_adapter_model_path, device=self.device)
        logging.info(f'init tex server done with cfg_json {self.cfg_json}')

    def load_cfg(self, cfg_json, model_name=None):
        """set self.model_name_list and self.cfg_model from json

        Args:
            cfg_json: relative name in codedir/configs
            model_name: model key. Defaults to None.
        """
        logging.info(f"input of load_cfg: cfg_json: {cfg_json}, model_name:{model_name}")
        self.configs_dir = os.path.join(codedir, 'configs')
        cfg_json_ = os.path.join(self.configs_dir, cfg_json)
        if not os.path.exists(cfg_json_):
            cfg_json_ = os.path.join(self.configs_dir, "tex_imguv.json")
        assert os.path.exists(cfg_json_), f"can not find valid cfg_json: {cfg_json}"
        cfg_root = load_json(cfg_json_) #  
        assert len(cfg_root), f"invalid cfg_json: {cfg_json}"

        model_name_list= list(cfg_root.keys())
        if model_name is not None and model_name in cfg_root:
            cfg_model = cfg_root[model_name]
        else:
            logging.warn(f"invalid model_name in cfg_json: {cfg_json}, model_name:{model_name}, juet select first")
            model_name = model_name_list[0]
            cfg_model = cfg_root[model_name]

        self.model_name_list = model_name_list
        self.cfg_model = edict(cfg_model)
        self.cfg_json = cfg_json_
        self.model_name = model_name
        logging.info(f"load_cfg done with self.model_name:{self.model_name}")
        return

    # common utils
    def get_job_id(self):
        """Get unique job ID after increasing the self.job_cnt

        Returns:
            int: newest job_id
        """
        self.lock.acquire()

        try:
            self.job_cnt += 1
        finally:
            self.lock.release()

        return self.job_cnt

    def finish_job(self, job_id, results):
        """Save the status and results in self.jobs

        Args:
            job_id (int): the unique ID of the job
            results (list[str]): the generated model paths
        """
        self.lock.acquire()

        self.jobs[job_id]['results'] = results
        self.jobs[job_id]['status'] = teximguv_pb2.FINISHED

        self.lock.release()

    def start_new_job(self, function, *args):
        """Start a new job and record everything in self.jobs

        Args:
            function (_type_): the function to run
            *args: the parameters pass to function

        Returns:
            int: job_id
        """
        job_id = self.get_job_id()

        self.lock.acquire()
        self.jobs[job_id] = {
            'status': teximguv_pb2.IN_PROGRESS,
        }
        self.lock.release()

        # print('start thread')
        job = threading.Thread(target=function, args=(job_id, *args))
        print('debug: thread started')
        job.start()

        self.lock.acquire()
        self.jobs[job_id]['job'] = job
        self.lock.release()

        return job_id

    def warp_run_cfg(
        self, uv_res=512, num_inference_steps=20, guidance_scale=7.5, debug_save=True
    ):
        cfg = {
            "uv_res": uv_res,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "debug_save": debug_save,
        }
        run_cfg = edict(cfg)

        return run_cfg

    # core functions to run pipeline
    def run_tex_imguv(self, job_id, in_obj, out_objs_dir, in_prompts=None, in_condi_img=None, 
                      uv_res=512, num_inference_steps=20, guidance_scale=7.5, debug_save=True):
        """feed text and image, get uv texture from obj
        Args:
            in_obj: raw obj path with uv coord
            out_objs_dir: output dir with multi output objs
            in_prompts: input text, list of text or None
            in_condi_img: condi img path or None
            out_debug_dir: save debug result if not None. Defaults to 'None'.

        Raises:
            e: _description_
        """
        logging.info(f'run_tex_imguv input in_obj=\"{in_obj}\", in_prompts={in_prompts} in_condi_img= {in_condi_img}')
        for i in range(0, self.retry):
            try:
                run_cfg = self.warp_run_cfg(uv_res, num_inference_steps, guidance_scale, debug_save)
                logging.info("run_tex_imguv run_cfg ", run_cfg)
                # return out_obj_paths or ''
                results = self.pipeline.interface_obj_teximguv(in_obj, out_objs_dir, in_prompts=in_prompts, 
                                                     in_condi_img=in_condi_img, run_cfg=run_cfg)
                if isinstance(results, list):
                    results = ",".join(results)
                break
            except Exception as e:
                logging.error(f'run_tex_imguv failed, retry={i}')
                if i < (self.retry - 1):
                    # TODO(csz) how to re-connect
                    pass

                else:
                    raise e

        logging.info(f'run_tex_imguv  done, in_obj=\"{in_obj}\", in_prompts={in_prompts} in_condi_img= {in_condi_img}"')
        self.finish_job(job_id, results)
        return

    # grpc protos interface functions
    def NewJob(self, request, context):
        """Check the inputs from request and start the AIGC job accordingly

        Args:
            request (JobRequest): the request object contains all the possible parameters
            context : gRPC context

        Returns:
            JobReply: The job status
        """
        print('debug request', request)
        
        def warp_string(in_string, magic_key=":::"):
            """_summary_

            Args:
                in_string: (merged with magic key :::) or ''
                magic_key: _description_. Defaults to ":::".

            Returns:
                _description_
            """
            if len(in_string) <= 0:
                return None
            text_list = in_string.split(magic_key)
            if len(text_list) == 1:
                return text_list[0]
            return text_list
            

        if request.task_type == teximguv_pb2.TEX_IMGUV:
            if len(request.in_obj) > 0:

                job_id = self.start_new_job(
                    self.run_tex_imguv,
                    request.in_obj,
                    request.out_objs_dir,
                    warp_string(request.in_prompts),
                    warp_string(request.in_condi_img),
                    request.uv_res,
                    request.num_inference_steps,
                    request.guidance_scale,
                    request.debug_save,
                )
            else:
                # TODO
                return teximguv_pb2.JobReply(job_id=-1, error="Invalid parameters")
        else:
            return teximguv_pb2.JobReply(job_id=-1, error="Invalid task_type")

        return teximguv_pb2.JobReply(job_id=job_id)

    def QueryJobs(self, request, context):
        """Query the status of jobs using list of job id

        Args:
            request (JobStatusRequest): the request that contains a list of job IDs

        Returns:
            JobStatusReply: the reply that contains a list of job JobReply
        """
        rets = []
        for job_id in request.job_ids:
            if job_id in self.jobs:
                info = self.jobs[job_id]
                # print('debug query job info', info)
                ret = teximguv_pb2.JobReply(
                    job_id=job_id,
                    status=info['status'],
                    estimated_time=10,
                    results=info.get('results', ''),
                    error='')
                rets.append(ret)

        return teximguv_pb2.JobStatusReply(job_status=rets)

    def Add(self, request, context):
        """simple function for debug
        """
        return teximguv_pb2.AdditionResponse(result=request.x + request.y)


def main_server(lj_port='8080', cfg_json='tex_imguv.json', model_key='mcwy'):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    teximguv_pb2_grpc.add_TeximguvServicer_to_server(TeximguvServer(cfg_json, model_key), server)
    server.add_insecure_port(f"[::]:{lj_port}")
    server.start()
    logging.info(f"Server started, listening on {lj_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('--lj_port', type=str, default='8080')
    parser.add_argument('--cfg_json', type=str, default='tex_imguv.json')
    parser.add_argument('--model_key', type=str, default='mcwy',
                        help='select model. can be mcwy, lowpoly')
    args = parser.parse_args()

    logging.basicConfig()
    main_server(args.lj_port, args.cfg_json, args.model_key)
