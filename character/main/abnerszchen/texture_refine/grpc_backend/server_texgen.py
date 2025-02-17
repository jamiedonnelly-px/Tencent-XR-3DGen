import logging
import argparse
import os
import sys
import grpc
import texgen_pb2
import texgen_pb2_grpc
from concurrent import futures
import concurrent.futures
import threading
from easydict import EasyDict as edict
import time
import threading
import subprocess
import time
import socket
import uuid

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from dataset.utils_dataset import load_json
from pipe_texuv import ObjTexUVPipeline   # SDXL
from pipe_texcontrol import ObjTexControlPipeline   # SD
from pipe_teximguv import ObjTexImgUVPipeline   # SD, train with image. uselss now

from client_blendercvt import BlenderCVTClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def is_grpc_server_running(host, port):
    try:
        with socket.create_connection((host, port), timeout=1) as sock:
            return True
    except (socket.timeout, ConnectionRefusedError, socket.error):
        return False

def cvt_glb_to_obj(in_glb_path, output_obj_path, blender_root="/usr/blender-3.6.2-linux-x64/blender"):
    """convert glb to obj with blender

    Args:
        in_glb_path: x.glb
        output_obj_path: xxx.obj
        blender_root: _description_. Defaults to "/usr/blender-3.6.2-linux-x64/blender".

    Returns:
        True if cvt done, else False
    """
    if not os.path.exists(in_glb_path):
        return False
    assert os.path.exists(blender_root), f"can not find blender_root {blender_root}"
    os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
    cmd = f"{blender_root} -b -P {codedir}/dataset/control_pre/glb_to_obj.py -- --mesh_path '{in_glb_path}' --output_obj_path '{output_obj_path}'"
    print('debug cmd ', cmd)
    subprocess.run(cmd, shell=True)
    return os.path.exists(output_obj_path)

def cvt_obj_to_glb(in_obj_path, out_glb_path, blender_root="/usr/blender-3.6.2-linux-x64/blender"):
    """convert obj to glb with blender

    Args:
        in_obj_path: x.obj
        out_glb_path: xx.glb
        blender_root: _description_. Defaults to "/usr/blender-3.6.2-linux-x64/blender".

    Returns:
        True if cvt done, else False
    """
    if not os.path.exists(in_obj_path):
        return False
    os.makedirs(os.path.dirname(out_glb_path), exist_ok=True)
    cmd = f"{blender_root} -b -P {codedir}/dataset/control_pre/obj_to_glb.py -- --input_path '{in_obj_path}' --output_glb_path '{out_glb_path}'"
    subprocess.run(cmd, shell=True)
    return os.path.exists(out_glb_path)


def log_time_list(time_list):
    if not time_list:
        return
    print('time_list ', time_list)
    prev_timestamp = time_list[0][1]
    time_sum = 0
    for name, time_stp in time_list:
        stage_duration = time_stp - prev_timestamp
        time_sum += stage_duration
        print("{} stage duration: {:.2f} seconds".format(name, stage_duration))
        prev_timestamp = time_stp
    print('time_sum ', time_sum)

    return

class TexGenServer(texgen_pb2_grpc.TexgenServicer):
    def __init__(self, cfg_json='tex_gen.json', model_name='uv_mcwy', max_run=1, device='cuda', max_finished_jobs=1000):
        """server

        Args:
            cfg_json: relative name in codedir/configs. Defaults to 'tex_gen.json'.
            model_name: key of cfg, can be selected in webui
            max_run: max_run same time
        """
        super(TexGenServer, self).__init__()
        self.load_cfg(cfg_json, model_name=model_name)
        self.device = device
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_run)
        self.max_finished_jobs = max_finished_jobs

        self.retry = 1  # Quick return error

        self.log_dir = os.path.join(self.cfg_model.log_root_dir, time.strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(self.log_dir, exist_ok=True)

        cvt_ip, cvt_port = "localhost", "987"
        self.use_blender_server = is_grpc_server_running(cvt_ip, cvt_port)
        if self.use_blender_server:
            self.blender_cvt_client = BlenderCVTClient(ip_port=f"{cvt_ip}:{cvt_port}")
        logging.info(f"use_blender_server: {self.use_blender_server}")

        logging.info(f'begin loading {self.model_name} model from {self.cfg_model.in_model_path}, Need a little time in T10')
        self.load_pipeline()

        self.task_pipe_type_map = {
            texgen_pb2.TEX_UV: "tex_uv",
            texgen_pb2.TEX_CONTROL: "tex_control",
            texgen_pb2.TEX_IMGUV: "tex_imguv",
        }

        logging.info(f'init tex server done with cfg_json {self.cfg_json}')

    def load_cfg(self, cfg_json, model_name=None):
        """set self.model_name_list and self.cfg_model from json, self.cfg_model=json[model_name]

        Args:
            cfg_json: relative name in codedir/configs
            model_name: model key. Defaults to None.
        """
        logging.info(f"input of load_cfg: cfg_json: {cfg_json}, model_name:{model_name}")
        self.configs_dir = os.path.join(codedir, 'configs')
        cfg_json_ = os.path.join(self.configs_dir, cfg_json)
        if not os.path.exists(cfg_json_):
            cfg_json_ = os.path.join(self.configs_dir, "tex_gen.json")
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
        logging.info(f"self.cfg_model:{self.cfg_model}")
        return

    def load_pipeline(self):
        """load our pipelien based on pipe_type in cfg json

        Raises:
            NotImplementedError: _description_
        """
        if self.cfg_model.pipe_type == "tex_uv":
            self.pipeline = ObjTexUVPipeline(
                self.cfg_model.in_model_path,
                self.cfg_model.in_sd_path,
                self.cfg_model.pretrained_vae_model_name_or_path if "pretrained_vae_model_name_or_path" in self.cfg_model else None,
                self.cfg_model.ip_adapter_model_path,
                device=self.device,
            )
        elif self.cfg_model.pipe_type == "tex_control":
            self.pipeline = ObjTexControlPipeline(
                self.cfg_model.in_model_path,
                self.cfg_model.in_sd_path,
                self.cfg_model.ip_adapter_model_path,
                device=self.device,
            )
        elif self.cfg_model.pipe_type == "tex_imguv":
            self.pipeline = ObjTexImgUVPipeline(
                self.cfg_model.in_model_path,
                self.cfg_model.in_sd_path,
                self.cfg_model.ip_adapter_model_path,
                device=self.device,
            )
        else:
            raise NotImplementedError(f"invalid pipe_type {self.cfg_model.pipe_type}")
        return

    ## ------------ common utils ---------------
    def wrap_run_cfg(
        self,
        uv_res=1024,
        num_inference_steps=20,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        debug_save=True,
    ):
        if self.cfg_model.pipe_type == "tex_uv":
            uv_res = 1024  # TODO
        else:
            uv_res = 512  # TODO
        cfg = {
            "uv_res": uv_res,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "ip_adapter_scale": ip_adapter_scale,
            "debug_save": debug_save,
        }
        run_cfg = edict(cfg)

        return run_cfg

    # ------------ grpc protos interface functions ------------
    def NewJob(self, request, context):
        """submit job thread, call self.NewJobHelper

        Args:
            request (JobRequest): the request object contains all the possible parameters
            context : gRPC context

        Returns:
            JobReply: The job status and results
        """
        # job_id from request or use default
        job_id = request.job_id
        if not job_id or job_id == "":
            print('[Warn] cannot get valid job_id, init it with uuid')
            job_id = str(uuid.uuid4())


        logging.debug(f"submit job_id: {job_id}")
        future = self.executor.submit(self.NewJobHelper, job_id, request)
        with self.jobs_lock:
            self.jobs[job_id] = {"status": texgen_pb2.QUEUED, "results": ""}
        
        result = future.result()  # will block until run done
        logging.debug(f"server run done job_id: {job_id} with result {result}")

        self.clear_old_jobs()
        job = self.jobs[job_id]

        reply = texgen_pb2.JobReply(job_id=job_id,
                                    status=job["status"],
                                    estimated_time=(job.get("end_time", 0) - job.get("sub_time", 0)),
                                    results=job.get("results", ""),
                                    error='')
        return reply

    def NewJobHelper(self, job_id, request):
        """run one thread

        Args:
            job_id: uuid from request
            request: with in_mesh_path, out_objs_dir,in_prompts, in_condi_img and params.. from JobRequest

        Returns:
            results
        """
        print('debug request', request)

        #### begin
        results = ""
        with self.jobs_lock:
            self.jobs[job_id] = {}
            self.jobs[job_id]["status"] = texgen_pb2.IN_PROGRESS
            self.jobs[job_id]["sub_time"] = time.time()

        #### core func
        if request.task_type in self.task_pipe_type_map:
            pipe_type = self.task_pipe_type_map[request.task_type]
            results = self.run_tex_gen(job_id, pipe_type,
                request.in_mesh_path,
                request.out_objs_dir,
                self.wrap_request_string(request.in_prompts),
                self.wrap_request_string(request.in_condi_img),
                self.wrap_request_string(request.in_obj_type),
                request.out_mesh_format,
                request.uv_res,
                request.num_inference_steps,
                request.guidance_scale,
                request.controlnet_conditioning_scale,
                request.ip_adapter_scale,
                request.debug_save)
            if not results:
                logging.error(f"[ERROR] run_tex_gen failed {results}")
                status = texgen_pb2.FAILED
            else:
                status = texgen_pb2.FINISHED
        else:
            logging.error(f"[ERROR] Invalid task_type {request.task_type}")
            status = texgen_pb2.FAILED

        #### end
        with self.jobs_lock:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["results"] = results
            self.jobs[job_id]["end_time"] = time.time()

        return results

    def QueryJobStatus(self, request, context):
        """Query the status of jobs using list of job id

        Args:
            request (JobStatusRequest): the request that contains a list of job IDs

        Returns:
            JobStatusReply: the reply status
            status, =2 when running:
                FINISHED      = 0; 
                FAILED        = 1;
                IN_PROGRESS   = 2;
                QUEUED   = 3;            
        """
        q_job_id = request.job_id
        with self.jobs_lock:
            if not q_job_id or not q_job_id in self.jobs:
                logging.error(f"[ERROR] invalid query job_id {q_job_id}")
                return texgen_pb2.JobStatusReply(status=texgen_pb2.FAILED)
            return texgen_pb2.JobStatusReply(status=self.jobs[q_job_id]["status"])
        
    def QueryServerStatus(self, request, context):
        """Query the thread cnt running now

        Args:
            request: _description_
            context: _description_

        Returns:
            ServerStatusReply with run_cnt, >0 means busy
        """
        run_job_ids = []
        with self.jobs_lock:
            for job_id, job_info in self.jobs.items():
                if job_info["status"] == texgen_pb2.IN_PROGRESS:
                    run_job_ids.append(job_id)
    
        return texgen_pb2.ServerStatusReply(run_cnt=len(run_job_ids))
    
    def Add(self, request, context):
        """simple function for debug
        """
        return texgen_pb2.AdditionResponse(result=request.x + request.y)


    #### debug
    def run_func(self, job_id, in_mesh, out_glb):
        with self.jobs_lock:
            self.jobs[job_id] = {}
            self.jobs[job_id]["status"] = texgen_pb2.IN_PROGRESS
        logging.info(f"begin run {job_id}")
        time.sleep(10)  # Simulate a long-running operation
        with self.jobs_lock:
            self.jobs[job_id]["status"] = texgen_pb2.FINISHED
        return "success"

    def RunFunc(self, request, context):
        job_id = str(uuid.uuid4())
        future = self.executor.submit(self.run_func, job_id, request.in_mesh, request.out_glb)
        logging.info(f"submit done")
        with self.jobs_lock:
            self.jobs[job_id] = {"status": texgen_pb2.QUEUED, "future": future}
        result = future.result()
        logging.info(f"run done")
        return texgen_pb2.RunFuncResponse(job_id=job_id, result=result)

    def QueryRunState(self, request, context):
        with self.jobs_lock:
            job_data = []
            for job_id, job_info in self.jobs.items():
                job_data.append(texgen_pb2.QueryRunStateResponse.JobData(job_id=job_id, status=job_info["status"]))
        return texgen_pb2.QueryRunStateResponse(jobs=job_data)
    def DebugNewJobHelper(self, job_id, request):
        with self.jobs_lock:
            self.jobs[job_id] = {}
            self.jobs[job_id]["status"] = texgen_pb2.IN_PROGRESS
            self.jobs[job_id]["sub_time"] = time.time()
        results = ""
        time.sleep(5)      
        with self.jobs_lock:
            self.jobs[job_id]["status"] = texgen_pb2.FINISHED
            self.jobs[job_id]["results"] = results
            self.jobs[job_id]["end_time"] = time.time()
        
        return results
    #### debug

    def clear_old_jobs(self):
        if len(self.jobs) < self.max_finished_jobs:
            return
        print('debug clear_old_jobs')
        finished_jobs = [jid for jid, job in self.jobs.items() if job["status"] in (texgen_pb2.FINISHED, texgen_pb2.FAILED)]
        if len(finished_jobs) > self.max_finished_jobs:
            oldest_finished_job_id = min(finished_jobs, key=lambda jid: self.jobs[jid]["end_time"])
            del self.jobs[oldest_finished_job_id]
        return

    ## ------------ core functions to run each pipeline ---------------
    def run_tex_gen(
        self,
        job_id,
        pipe_type,
        in_mesh_path,
        out_objs_dir,
        in_prompts=None,
        in_condi_img=None,
        in_obj_type=None,
        out_mesh_format="glb",
        uv_res=1024,
        num_inference_steps=20,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        debug_save=True,
    ):
        """feed text and image, get uv texture from mesh, results is out_mesh_paths(list of {out_mesh_format} path) or ""

        Args:
            job_id: uuid from request
            pipe_type: tex_control or tex_imguv
            in_mesh_path: raw mesh path with uv coord, can be obj or glb
            out_objs_dir: output dir with multi output objs
            in_prompts: input text, list of text or None
            in_condi_img: condi img path or None
            in_obj_type: obj type(top, dress, and so on) or None
            out_mesh_format: output mesh as glb or obj
            uv_res: generate uv resolution
            num_inference_steps: diffusion loop steps
            guidance_scale: text guidance scale:The larger the value, the closer it is to the text prompt. Defaults to 9.0.
            controlnet_conditioning_scale (int, optional): The higher the value, the better the control but the worse the quality. in [0, 1]
            ip_adapter_scale (int, optional): The larger the value, the closer it is to the image. in [0, 1]
            debug_save: save debug result if True.
        Return:
            results(string): ",".join(list of out_obj_paths)
        Raises:
            e: _description_
        """
        log_name = f"run_{pipe_type}"
        logging.info(f'{log_name} input in_mesh_path=\"{in_mesh_path}\", in_prompts={in_prompts} in_condi_img= {in_condi_img}')
        logging.info(f'job_id: {job_id}')
        results = ""
        for i in range(0, self.retry):
            try:
                time_list = []
                time_list.append(('start', time.time()))

                run_cfg = self.wrap_run_cfg(
                    uv_res,
                    num_inference_steps,
                    guidance_scale,
                    controlnet_conditioning_scale,
                    ip_adapter_scale,
                    debug_save,
                )
                logging.info(f"{log_name} run_cfg ", run_cfg)
                print(f"[run_tex_gen] {log_name} run_cfg ", run_cfg)

                in_obj = self.wrap_in_mesh_path(in_mesh_path)
                time_list.append(('wrap_in_done', time.time()))

                ## TODO(csz) merge
                ## results return out_mesh_paths or ''
                if pipe_type == "tex_uv":
                    assert self.pipeline.pipe_type == "tex_uv" and isinstance(self.pipeline, ObjTexUVPipeline)
                    results = self.pipeline.interface_obj_texuv(
                        in_obj,
                        out_objs_dir,
                        in_prompts=in_prompts,
                        in_condi_img=in_condi_img,
                        run_cfg=run_cfg,
                    )
                elif pipe_type == "tex_control":
                    assert self.pipeline.pipe_type == "tex_control" and isinstance(self.pipeline, ObjTexControlPipeline)
                    results = self.pipeline.interface_obj_texcontrol(
                        in_obj,
                        out_objs_dir,
                        in_prompts=in_prompts,
                        in_condi_img=in_condi_img,
                        run_cfg=run_cfg,
                    )
                elif pipe_type == "tex_imguv":
                    assert self.pipeline.pipe_type == "tex_imguv" and isinstance(self.pipeline, ObjTexImgUVPipeline)
                    results = self.pipeline.interface_obj_teximguv(
                        in_obj,
                        out_objs_dir,
                        in_prompts=in_prompts,
                        in_condi_img=in_condi_img,
                        run_cfg=run_cfg,
                    )

                time_list.append(('interface_obj_done', time.time()))

                if out_mesh_format == "glb":
                    results = self.wrap_out_obj_paths(results)

                time_list.append(('wrap_out__done', time.time()))
                try:
                    log_time_list(time_list)
                except Exception as e:
                    print('log_time_list failed. skip', e)

                # merge results as one string for grpc output
                if isinstance(results, list):
                    results = ",".join(results)
                break
            except Exception as e:
                logging.error(f'{log_name} failed, retry={i}')
                if i < (self.retry - 1):
                    # TODO(csz) how to re-connect
                    pass

                else:
                    break
                    # raise e

        logging.info(
            f'{log_name}  done, in_mesh_path="{in_mesh_path}", in_prompts={in_prompts} in_condi_img= {in_condi_img}", results={results}'
        )

        return results

    def wrap_request_string(self, in_string, magic_key=":::"):
        """convert request string to valid string/list of string/None

        Args:
            in_string: (merged with magic key :::) or ''
            magic_key: _description_. Defaults to ":::".

        Returns:
            return None if is '', return string or list of string
        """
        if len(in_string) <= 0:
            return None
        text_list = in_string.split(magic_key)
        if len(text_list) == 1:
            return text_list[0]
        return text_list

    def wrap_in_mesh_path(self, in_mesh_path):
        """convert in mesh to obj

        Args:
            in_mesh_path: obj or glb path

        Returns:
            obj path
        """
        if not in_mesh_path or not os.path.exists(in_mesh_path):
            raise ValueError(f"invalid in_mesh_path: {in_mesh_path}")
        filename, file_extension = os.path.splitext(in_mesh_path)
        if file_extension == ".glb":
            output_obj_path = os.path.join(
                os.path.dirname(in_mesh_path), filename, "cvt_obj/mesh.obj"
            )
            if self.use_blender_server:
                flag = self.blender_cvt_client.interface_glb_to_obj(in_mesh_path, output_obj_path)
            else:
                flag = cvt_glb_to_obj(in_mesh_path, output_obj_path)
            if not flag:
                raise ValueError(
                    f"cvt glb to obj failed! use_blender_server:{self.use_blender_server}"
                )
            in_obj = output_obj_path
        elif file_extension == ".obj":
            in_obj = in_mesh_path
        else:
            raise ValueError(f"invalid in_mesh format: {in_mesh_path}")
        return in_obj

    def wrap_out_obj_paths(self, out_obj_paths, out_type="glb"):
        """convert out objs to glbs

        Args:
            out_obj_paths: "" or list of obj path
            out_type: _description_. Defaults to "glb".

        Raises:
            ValueError: _description_

        Returns:
            "" or list of {out_type} path
        """
        if not out_obj_paths:
            return out_obj_paths

        if out_type == "glb":
            out_mesh_paths = []
            for obj_path in out_obj_paths:
                _, file_extension = os.path.splitext(obj_path)
                if file_extension == ".glb":
                    out_mesh_paths.append(out_glb_path)
                elif file_extension == ".obj":
                    out_glb_path = obj_path.replace('.obj', '.glb')
                    if self.use_blender_server:
                        flag = self.blender_cvt_client.interface_obj_to_glb(obj_path, out_glb_path)
                    else:
                        flag = cvt_obj_to_glb(obj_path, out_glb_path)
                    if not flag:
                        raise ValueError("cvt_obj_to_glb failed")
                    out_mesh_paths.append(out_glb_path)
                else:
                    continue
            return out_mesh_paths
        else:
            return out_obj_paths


def main_server(lj_port='8986', cfg_json='tex_gen.json', model_key='uv_mcwy', max_workers=12):
    print("input", lj_port, cfg_json, model_key)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    texgen_pb2_grpc.add_TexgenServicer_to_server(TexGenServer(cfg_json, model_key, max_run=1), server)
    server.add_insecure_port(f"[::]:{lj_port}")
    server.start()
    logging.info(f"Server started, listening on {lj_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('--lj_port', type=str, default='8986')
    parser.add_argument('--cfg_json', type=str, default='tex_gen.json')
    parser.add_argument('--model_key', type=str, default='uv_mcwy',
                        help='select model. can be uv_mcwy, control_mcwy, imguv_mcwy, imguv_lowpoly, pipe_type_dataset')
    args = parser.parse_args()

    logging.basicConfig()
    main_server(args.lj_port, args.cfg_json, args.model_key)
