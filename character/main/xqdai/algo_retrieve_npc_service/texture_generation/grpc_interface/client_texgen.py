import time
import os
import json
import uuid
import argparse
import logging
import threading
import grpc

import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import texgen_pb2
import texgen_pb2_grpc
from easydict import EasyDict as edict
import concurrent.futures
import uuid

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


### utils func
def load_json(in_file):
    with open(in_file, encoding="utf-8") as f:
        data = json.load(f)
    return data


def init_job_id():
    return str(uuid.uuid4())


def wrap_output(out_mesh_paths_queryed):
    """convert ",".join([mesh]) from grpc to [mesh] list

    Args:
        out_mesh_paths_queryed: If '' is returned, the task failed. If run success, it is ",".join(out_mesh_paths)
    Return list of obj path
    """
    if not out_mesh_paths_queryed:
        return [""]
    out_mesh_paths = out_mesh_paths_queryed.split(",")

    return out_mesh_paths


def query_mesh_from_key(web_flatten_dict, in_mesh_path, in_mesh_key):
    """find mesh from web_flatten_dict

    Args:
        web_flatten_dict: Example: /aigc_cfs_3/layer_tex/mcwy/merge/web_0406/web_flatten.json
        in_mesh_path: raw mesh path
        in_mesh_key: obj key or None

    Returns:
        _description_
    """
    try:
        if in_mesh_key is None or (in_mesh_key not in web_flatten_dict):
            print(f"debug invalid in_mesh_key {in_mesh_key}, keep raw input")
            return in_mesh_path

        Mesh_obj_raw = web_flatten_dict[in_mesh_key]["Mesh_obj_raw"]
        if os.path.exists(Mesh_obj_raw):
            print(
                f"debug find valid mesh {Mesh_obj_raw } from in_mesh_key {in_mesh_key}"
            )
            return Mesh_obj_raw
        else:
            print(f"debug have in_mesh_key {in_mesh_key} but can not find exists mesh")
            return in_mesh_path
    except Exception as e:
        print("error when query_mesh_from_key, keep raw ", e)
        return in_mesh_path


# main class
class TexGenClient:
    def __init__(
        self, client_cfg_json="client_texgen.json", model_name="uv_mcwy", device="cuda"
    ):
        """client

        Args:
            client_cfg_json: abs path or relative name in codedir/configs. Defaults to 'client_texgen.json'.
            model_name: key of cfg, can be selected in webui
            device
        """
        self.load_run_cfg(client_cfg_json, model_name=model_name)
        self.device = device
        self.lock = threading.Lock()
    
        self.run_cfg.server_addr

        self.status_vis_map = {
            texgen_pb2.FINISHED: "finished",
            texgen_pb2.FAILED: "failed",
            texgen_pb2.IN_PROGRESS: "in_progress",
            texgen_pb2.QUEUED: "queued",
        }

        try:
            if not os.path.exists(self.run_cfg.web_flatten_json):
                # when use relative path
                self.web_flatten_json = os.path.join(
                    os.path.dirname(self.cfg_json), self.run_cfg.web_flatten_json
                )
            else:
                self.web_flatten_json = (
                    self.run_cfg.web_flatten_json
                )  # when use abs path
            assert os.path.exists(
                self.web_flatten_json
            ), f"can not find web_flatten_json: {self.web_flatten_json}"
            self.web_flatten_dict = load_json(self.web_flatten_json)
            logging.info(
                f"load web_flatten_json done from {self.web_flatten_json}, find {len(self.web_flatten_dict)} objs"
            )

        except Exception as e:
            print("load web_flatten_json failed", e)
            raise ValueError("load web_flatten_json failed")

    def reset(self):
        # TODO
        return

    def load_run_cfg(self, cfg_json, model_name=None):
        """set self.model_name_list and self.run_cfg from json

        Args:
            cfg_json: abs path or relative name in codedir/configs
            model_name: model key. Defaults to None.
        """
        logging.info(
            f"input of load_run_cfg: cfg_json: {cfg_json}, model_name:{model_name}"
        )
        if os.path.exists(cfg_json):
            cfg_json_ = os.path.abspath(cfg_json)
        else:
            self.configs_dir = os.path.join(codedir, "configs")
            cfg_json_ = os.path.join(self.configs_dir, cfg_json)
            if not os.path.exists(cfg_json_):
                cfg_json_ = os.path.join(self.configs_dir, "client_texgen.json")
        assert os.path.exists(
            cfg_json_
        ), f"can not find valid cfg_json: {cfg_json}, {cfg_json_}"
        cfg_root = load_json(cfg_json_)  #
        assert len(cfg_root), f"invalid cfg_json: {cfg_json}"

        model_name_list = []
        for model_name_, cfg_ in cfg_root.items():
            if cfg_.get("active", False):
                model_name_list.append(model_name_)
        if not model_name_list or len(model_name_list) < 1:
            raise ValueError(f"No active model in cligen cfg {cfg_json_}")

        if model_name is not None and model_name in cfg_root:
            run_cfg = cfg_root[model_name]
        else:
            logging.warn(
                f"invalid model_name in cfg_json: {cfg_json}, model_name:{model_name}, juet select first"
            )
            model_name = model_name_list[0]
            run_cfg = cfg_root[model_name]

        self.model_name_list = model_name_list
        self.run_cfg = edict(run_cfg)
        self.cfg_json = cfg_json_
        self.model_name = model_name

        self.pipe_task_type_map = {
            "tex_uv": texgen_pb2.TEX_UV,
            "tex_control": texgen_pb2.TEX_CONTROL,
            "tex_imguv": texgen_pb2.TEX_IMGUV,
        }

        logging.info(
            f"load_run_cfg done with self.model_name:{self.model_name} with input model_name {model_name}"
        )
        return

    ### core function of grpc
    def client_obj_tex_gen(
        self,
        job_id,
        pipe_type,
        in_mesh_path,
        in_mesh_key=None,
        out_objs_dir=None,
        in_prompts="",
        in_condi_img="",
        in_obj_type="",
        out_mesh_format="glb",
        uv_res=1024,
        num_inference_steps=20,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.7,
        ip_adapter_scale=0.7,
        debug_save=True,
    ):
        """query grpc server of tex imguv

        Args:
            job_id(string): uuid
            pipe_type(string): need be tex_uv, tex_control or tex_imguv, select pipeline
            in_mesh_path(string): raw obj/glb path with uv coord
            in_mesh_key(string): query mesh obj from web_flatten_dict
            out_objs_dir(string): output dir with multi output objs, if is None, use default dir(cfg.log_root_dir)
            in_prompts(string): input text, list of text (merged with magic key :::) or ''
            in_condi_img(string): condi img path or ''
            in_obj_type(string): obj_type or '' TODO(csz, not used now)
            out_mesh_format glb/obj
            uv_res 1024/512, xl use 1024
            num_inference_steps 20
            guidance_scale (int, optional): The larger the value, the closer it is to the text prompt. Defaults to 9.0.
            controlnet_conditioning_scale (int, optional): The higher the value, the better the control but the worse the quality. in [0, 1]
            ip_adapter_scale (int, optional): The larger the value, the closer it is to the image. in [0, 1]

        Returns:
            out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
        """
        server_addr = self.run_cfg.server_addr
        if out_objs_dir is None:
            log_root_dir = self.run_cfg.log_root_dir
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            out_objs_dir = os.path.join(log_root_dir, str(unique_id))
            os.makedirs(out_objs_dir, exist_ok=True)

        if not pipe_type or not pipe_type in self.pipe_task_type_map:
            print(f"[Failed] job_id {job_id} invalid pipe_type {pipe_type}")
            return [""]
        task_type = self.pipe_task_type_map[pipe_type]

        in_mesh_path = query_mesh_from_key(
            self.web_flatten_dict, in_mesh_path, in_mesh_key
        )
        logging.info(
            f"begin client_obj_tex_gen, job_id {job_id} with task_type {task_type} and in_mesh_key {in_mesh_key}, in_mesh_path {in_mesh_path}"
        )

        with grpc.insecure_channel(server_addr) as channel:
            stub = texgen_pb2_grpc.TexgenStub(channel)
            ### begin
            # refer data to protos/texgen.proto message JobRequest; func to NewJob in server request.xxx, with run request.run_tex_gen
            response = stub.NewJob(
                texgen_pb2.JobRequest(
                    job_id=job_id,
                    task_type=task_type,
                    in_mesh_path=in_mesh_path,
                    out_objs_dir=out_objs_dir,
                    in_prompts=in_prompts,
                    in_condi_img=in_condi_img,
                    in_obj_type=in_obj_type,
                    out_mesh_format=out_mesh_format,
                    uv_res=uv_res,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    ip_adapter_scale=ip_adapter_scale,
                    debug_save=debug_save,
                )
            )
            print("debug response ", response)
            job_id = response.job_id
            out_mesh_paths_queryed = response.results

            if not out_mesh_paths_queryed:
                print(f"[Failed] job_id {job_id} run failed")
                return [""]

            print(
                f"[Succeed], job_id {job_id} get out_mesh_paths_queryed:{out_mesh_paths_queryed}"
            )
            out_mesh_paths = wrap_output(out_mesh_paths_queryed)

        ### end
        nake_model = os.path.join(os.path.dirname(in_mesh_path), "nake_model")

        if os.path.exists(nake_model) and out_mesh_paths and len(out_mesh_paths) > 0:
            for out_mesh_path in out_mesh_paths:
                out_dir = os.path.dirname(out_mesh_path)
                os.system(f"cp -r {nake_model} {out_dir}")

        return out_mesh_paths

    def query_job_status(self, job_id):
        """Query the status of jobs using list of job id

        Args:
            job_id: _description_

        Returns:
            status, =2 when running:
                FINISHED      = 0;
                FAILED        = 1;
                IN_PROGRESS   = 2;
                QUEUED   = 3;
        """
        status = ""
        with grpc.insecure_channel(self.run_cfg.server_addr) as channel:
            stub = texgen_pb2_grpc.TexgenStub(channel)
            response = stub.QueryJobStatus(texgen_pb2.JobStatusRequest(job_id=job_id))
            status = response.status
            return self.status_vis_map.get(status, "unknown")

    def query_server_status(self):
        """Query the thread cnt running now

        Returns:
            run_cnt, >0 means busy
        """
        run_cnt = -1
        with grpc.insecure_channel(self.run_cfg.server_addr) as channel:
            stub = texgen_pb2_grpc.TexgenStub(channel)
            response = stub.QueryServerStatus(texgen_pb2.ServerStatusRequest())
            run_cnt = response.run_cnt
        return run_cnt

    ###
    def test_add(self):
        logging.info(f"run test_add at: {self.run_cfg.server_addr}")
        with grpc.insecure_channel(self.run_cfg.server_addr) as channel:
            stub = texgen_pb2_grpc.TexgenStub(channel)
            response = stub.Add(texgen_pb2.AdditionRequest(x=5, y=3))
            print("texgen 5 + 3 =", response.result)

    ### interface of web ui
    def webui_query_text(
        self,
        job_id,
        in_mesh_path,
        in_prompts,
        in_mesh_key=None,
        out_objs_dir=None,
        in_obj_type="",
        pipe_type="tex_uv",
    ):
        """webui query text only mode

        Args:
            job_id(string): uuid
            in_mesh_path(string): raw obj/glb path with uv coord
            in_prompts(string/ list of string): input text, list of text (merged with magic key :::) or ''
            in_mesh_key(string): query mesh obj from self.web_flatten_dict
            out_objs_dir(string): output dir with multi output objs, if None use default dir(cfg.log_root_dir)
            in_obj_type(string): obj_type or '' TODO(csz, not used now)
            pipe_type(string): need be tex_uv, tex_control or tex_imguv, select pipeline
        Returns:
            out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
        """
        out_mesh_paths = self.client_obj_tex_gen(
            job_id=job_id,
            pipe_type=pipe_type,
            in_mesh_path=in_mesh_path,
            in_mesh_key=in_mesh_key,
            out_objs_dir=out_objs_dir,
            in_prompts=in_prompts,
            in_condi_img="",
            in_obj_type=in_obj_type,
            guidance_scale=self.run_cfg.guidance_scale,
            controlnet_conditioning_scale=self.run_cfg.controlnet_conditioning_scale,
            ip_adapter_scale=self.run_cfg.ip_adapter_scale,
        )
        # self.run_cfg
        return out_mesh_paths

    def webui_query_image(
        self,
        job_id,
        in_mesh_path,
        in_condi_img,
        in_mesh_key=None,
        out_objs_dir=None,
        in_obj_type="",
        pipe_type="tex_uv",
    ):
        """webui query image only mode

        Args:
            job_id(string): uuid
            in_mesh_path(string): raw obj/glb path with uv coord
            in_condi_img(string): condi img path
            in_mesh_key(string): query mesh obj from self.web_flatten_dict
            out_objs_dir(string): output dir with multi output objs, if None use default dir(cfg.log_root_dir)
            in_obj_type(string): obj_type or '' TODO(csz, not used now)
            pipe_type(string): need be tex_uv, tex_control or tex_imguv, select pipeline
        Returns:
            out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
        """

        out_mesh_paths = self.client_obj_tex_gen(
            job_id=job_id,
            pipe_type=pipe_type,
            in_mesh_path=in_mesh_path,
            in_mesh_key=in_mesh_key,
            out_objs_dir=out_objs_dir,
            in_prompts="",
            in_condi_img=in_condi_img,
            in_obj_type=in_obj_type,
            guidance_scale=self.run_cfg.guidance_scale,
            controlnet_conditioning_scale=self.run_cfg.controlnet_conditioning_scale,
            ip_adapter_scale=self.run_cfg.ip_adapter_scale,
        )
        return out_mesh_paths

    def webui_query_text_image(
        self,
        job_id,
        in_mesh_path,
        in_prompts,
        in_condi_img,
        in_mesh_key=None,
        out_objs_dir=None,
        in_obj_type="",
        pipe_type="tex_uv",
    ):
        """webui query text+image mode

        Args:
            job_id(string): uuid
            in_mesh_path(string): raw obj/glb path with uv coord
            in_prompts(string/ list of string): input text, list of text (merged with magic key :::) or ''
            in_condi_img(string): condi img path
            in_mesh_key(string): query mesh obj from self.web_flatten_dict
            out_objs_dir(string): output dir with multi output objsoutput dir with multi output objs, if None use default dir(cfg.log_root_dir)
            in_obj_type(string): obj_type or '' TODO(csz, not used now)
            pipe_type(string): need be tex_uv, tex_control or tex_imguv, select pipeline
        Returns:
            out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
        """
        mix_ip_division_scale = (
            4.0
            if "mix_ip_division_scale" not in self.run_cfg
            else self.run_cfg.mix_ip_division_scale
        )
        out_mesh_paths = self.client_obj_tex_gen(
            job_id=job_id,
            pipe_type=pipe_type,
            in_mesh_path=in_mesh_path,
            in_mesh_key=in_mesh_key,
            out_objs_dir=out_objs_dir,
            in_prompts=in_prompts,
            in_condi_img=in_condi_img,
            in_obj_type=in_obj_type,
            guidance_scale=self.run_cfg.guidance_scale,
            controlnet_conditioning_scale=self.run_cfg.controlnet_conditioning_scale,
            ip_adapter_scale=self.run_cfg.ip_adapter_scale
            / mix_ip_division_scale,  # use small ip scale when mix mode
        )
        return out_mesh_paths


def thread_once(client):
    in_mesh_path = f"/aigc_cfs/Asset/designcenter/clothes/convert/mcwy2/remove_skin_mesh/meshes/Top/BR_TOP_1_F_T/BR_TOP_1_fbx2020.obj"
    # in_mesh_path = f"/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/MCWY_2_Dress/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/uv_condition/mesh.obj"
    out_objs_dir = f"/aigc_cfs_3/sz/server/tex_gen/client_log/mcwy_debug_thread"
    # in_prompts = ""
    in_condi_img = "/aigc_cfs/Asset/designcenter/clothes/render_part2/render_data/dress/render_data/DR_640_F_A/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/color/cam-0032.png"
    in_prompts = "indian style"
    out_mesh_paths_text = client.webui_query_text(
        in_mesh_path,
        in_prompts,
        in_mesh_key=None,
        out_objs_dir=os.path.join(out_objs_dir, "text"),
        in_obj_type="",
        pipe_type=client.run_cfg.pipe_type,
    )
    return


def run_func(stub, in_mesh, out_glb):
    response = stub.RunFunc(texgen_pb2.RunFuncRequest(in_mesh=in_mesh, out_glb=out_glb))
    print(f"RunFunc result: {response.result}, job_id: {response.job_id}")


def query_run_state(stub):
    response = stub.QueryRunState(texgen_pb2.QueryRunStateRequest())
    print("QueryRunState:")
    for job_data in response.jobs:
        logging.info(f"Job ID: {job_data.job_id}, Status: {job_data.status}")


def test_easy_thread(server_addr):
    with grpc.insecure_channel(server_addr) as channel:
        stub = texgen_pb2_grpc.TexgenStub(channel)

        # Run multiple tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(run_func, stub, "in_mesh", "out_glb") for _ in range(5)
            ]

            # Query the run state every second
            while not all(f.done() for f in futures):
                query_run_state(stub)
                time.sleep(1)

            # Wait for all run_func tasks to complete
            concurrent.futures.wait(futures)

        # Query the run state after all tasks are completed
        query_run_state(stub)


def test_client(client_cfg_json, model_key):
    client = TexGenClient(client_cfg_json, model_name=model_key)

    # test_easy_thread(client.run_cfg.server_addr)
    # return

    print(f"lj_ip_port: {client.run_cfg.server_addr}")
    print("model_key ", model_key)
    print("client.run_cfg ", client.run_cfg)
    print("client.model_name_list ", client.model_name_list)
    client.test_add()

    if model_key == "control_ready":
        in_mesh_path = os.path.join(codedir, f"grpc_backend/files/ready/top/123.glb")
        out_objs_dir = "/aigc_cfs_3/sz/server/tex_gen/client_log/ready_debug"
        # in_prompts = ""
        # in_condi_img = "/aigc_cfs/Asset/designcenter/clothes/render_part2/render_data/dress/render_data/DR_640_F_A/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/color/cam-0032.png"
        in_prompts = "red chinese dragon"
        in_condi_img = ""
        # out_debug_dir=os.path.dirname(out_obj)
    elif model_key == "uv_mcwy" or model_key == "control_mcwy":
        # in_mesh_path = f"/aigc_cfs_3/layer_tex/mcwy_2/2024/MCWY_2_Bottom/BR_BTM_5/uv_condition/mesh.obj"
        # in_mesh_path = f"/aigc_cfs_3/layer_tex/mcwy_2/manual_4class_0416/MCWY_2_Shoe/SH_164/uv_condition/mesh.obj"
        # in_mesh_path = f"/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/MCWY_2_Dress/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/uv_condition/mesh.obj"
        in_mesh_path = f"/aigc_cfs/Asset/designcenter/clothes/convert/mcwy2/remove_skin_mesh/meshes/Top/BR_TOP_1_F_T/BR_TOP_1_fbx2020.obj"
        out_objs_dir = (
            f"/aigc_cfs_3/sz/server/tex_gen/client_log/mcwy_debug_{model_key}"
        )
        # in_prompts = ""
        in_condi_img = "/aigc_cfs/Asset/designcenter/clothes/render_part2/render_data/dress/render_data/DR_640_F_A/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/color/cam-0032.png"
        in_prompts = "a yellow t shirt with a brown bear on it"
        # in_prompts = "White shoes with red spots on the side, HDR, UHD, 64K"
        # in_prompts = "indian style"
        # in_prompts = "red chinese dragon"
        # out_debug_dir=os.path.dirname(out_obj)
    elif model_key == "imguv_mcwy":
        in_mesh_path = f"/aigc_cfs/layer_avatar_data/mcwy_2/objs_three/MCWY_2_Dress/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/uv_condition/mesh.obj"
        out_objs_dir = "/aigc_cfs_3/sz/server/tex_imguv/client_log/mcwy_debug"
        in_prompts = ""
        in_condi_img = "/aigc_cfs/Asset/designcenter/clothes/render_part2/render_data/dress/render_data/DR_640_F_A/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/color/cam-0032.png"
        # out_debug_dir=os.path.dirname(out_obj)
    elif model_key == "imguv_lowpoly":
        # oname = '012c38ecb7f9308e805decd077adbef6c9d31af8_manifold_full_output_512_MightyWSB'
        # in_condi = f'/aigc_cfs_3/sz/data/tex/human/all_1222/Designcenter_1/{oname}/color/cam-0100.png'
        in_mesh_path = (
            "/aigc_cfs/sz/data/tex/lowpoly/low_poly/Dog_C_Police/uv_condition/mesh.obj"
        )
        out_objs_dir = "/aigc_cfs_3/sz/server/tex_imguv/client_log/lowploy_debug"
        # in_prompts = ''
        in_prompts = ":::".join(["", "low poly"])
        in_condi_img = "/aigc_cfs/Asset/artcenter/low_poly_srender/render_data/Dog_C_Police/Dog_C_Police_Dog_C_Police_manifold_full_output_512_MightyWSB/color/cam-0037.png"
        # out_obj = '/aigc_cfs_3/sz/result/tex_creator/human/pose8_argum/g8/design_2k_b16a2_nsddpm/new_objs_test_1_pipe/Designcenter_1/3k/obj_3k/xatlas/out/mesh.obj'

    else:
        print("invalid model_key")
        exit()

    ### test aigc webui func with query key
    # run once. *As an example of an external call*
    job_id = init_job_id()
    out_mesh_paths_query_key = client.webui_query_text(
        job_id,
        in_mesh_path,
        in_prompts,
        in_mesh_key="BR_TOP_1_F_T",
        out_objs_dir=os.path.join(out_objs_dir, "query_key"),
        in_obj_type="",
        pipe_type=client.run_cfg.pipe_type,
    )
    print("out_mesh_paths_query_key ", out_mesh_paths_query_key)
    job_status = ""
    job_status = client.query_job_status(job_id)
    run_cnt = client.query_server_status()
    print(
        f"external call example: job_id {job_id} job_status {job_status}, run_cnt {run_cnt}"
    )

    ### test aigc webui func with three interface
    out_mesh_paths_text = client.webui_query_text(
        init_job_id(),
        in_mesh_path,
        in_prompts,
        in_mesh_key=None,
        out_objs_dir=os.path.join(out_objs_dir, "text"),
        in_obj_type="",
        pipe_type=client.run_cfg.pipe_type,
    )
    print("out_mesh_paths_text ", out_mesh_paths_text)
    out_mesh_paths_image = client.webui_query_image(
        init_job_id(),
        in_mesh_path,
        in_condi_img,
        in_mesh_key=None,
        out_objs_dir=os.path.join(out_objs_dir, "image"),
        in_obj_type="",
        pipe_type=client.run_cfg.pipe_type,
    )
    print("out_mesh_paths_image ", out_mesh_paths_image)
    out_mesh_paths_mix = client.webui_query_text_image(
        init_job_id(),
        in_mesh_path,
        in_prompts,
        in_condi_img,
        in_mesh_key=None,
        out_objs_dir=None,
        in_obj_type="",
        pipe_type=client.run_cfg.pipe_type,
    )
    print("out_mesh_paths_mix ", out_mesh_paths_mix)

    #### just for my test
    debug_thread = True
    if not debug_thread:
        return

    # test common func with threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        job_ids, futures = [], []
        for i in range(3):
            job_id = init_job_id()
            job_ids.append(job_id)
            futures.append(
                executor.submit(
                    client.client_obj_tex_gen,
                    job_id=job_id,
                    pipe_type=client.run_cfg.pipe_type,
                    in_mesh_path=in_mesh_path,
                    out_objs_dir=os.path.join(out_objs_dir, f"thread_{i}"),
                    in_prompts=in_prompts,
                    in_condi_img=in_condi_img,
                    in_obj_type="",
                    out_mesh_format=client.run_cfg.out_mesh_format,
                    uv_res=client.run_cfg.uv_res,
                    num_inference_steps=client.run_cfg.num_inference_steps,
                    guidance_scale=client.run_cfg.guidance_scale,
                    controlnet_conditioning_scale=client.run_cfg.controlnet_conditioning_scale,
                    ip_adapter_scale=client.run_cfg.ip_adapter_scale,
                    debug_save=client.run_cfg.debug_save,
                )
            )

        print("debug submit done")

        # Query the run state every second
        while not all(f.done() for f in futures):
            for job_id in job_ids:
                job_status = client.query_job_status(job_id)
                logging.info(f"[QUERY] {job_id} job_status {job_status}")
            run_cnt = client.query_server_status()
            logging.info(f"[QUERY] run_cnt {run_cnt}")
            time.sleep(2)

        # Wait for all run_func tasks to complete
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="render est obj list")
    parser.add_argument(
        "--client_cfg_json",
        type=str,
        default="client_texgen.json",
        help="relative name in codedir/configs",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="uv_mcwy",
        help="select model. can be uv_mcwy, control_ready, control_mcwy, imguv_mcwy, imguv_lowpoly",
    )
    args = parser.parse_args()

    test_client(args.client_cfg_json, args.model_key)
