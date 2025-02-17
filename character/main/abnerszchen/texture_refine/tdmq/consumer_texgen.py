import time
import argparse
import logging
import pulsar
from _pulsar import ConsumerType, InitialPosition, LoggerLevel
from easydict import EasyDict as edict
import subprocess
import socket
import uuid
import json
import os
import sys
from PIL import Image
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.getLogger("pulsar").setLevel(logging.WARNING)

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from dataset.utils_dataset import load_json
from pipe_texuv import ObjTexUVPipeline  # SDXL
from pipe_texcontrol import ObjTexControlPipeline  # SD


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

def blender_add_image(in_obj_path, out_obj_path, blender_root="/usr/blender-3.6.2-linux-x64/blender"):
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
    os.makedirs(os.path.dirname(in_obj_path), exist_ok=True)
    cmd = f"{blender_root} -b -P {codedir}/dataset/control_pre/blender_remesh.py -- --mesh_path '{in_obj_path}' --output_mesh_path '{out_obj_path}' --process_stages 'add_image'"
    subprocess.run(cmd, shell=True)
    return os.path.exists(out_obj_path)


def log_time_list(time_list):
    try:
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
    except Exception as e:
        print(f'log_time_list failed, e={e} time_list=', time_list)
    return

def query_mesh_from_key(web_flatten_dict, in_mesh_path, in_mesh_key):
    """find mesh from web_flatten_dict, if find in_mesh_key use web_flatten_dict[in_mesh_key], else use in_mesh_path
    
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
            print(f"find valid {Mesh_obj_raw } from in_mesh_key {in_mesh_key}")
            return Mesh_obj_raw
        else:
            print(f"debug have in_mesh_key {in_mesh_key} but can not find exists mesh")
            return in_mesh_path
    except Exception as e:
        print("error when query_mesh_from_key, keep raw ", e)
        return in_mesh_path

def query_value_from_key(web_flatten_dict, in_mesh_key, query_key="uv_pos"):
    """find value geom_png/Category from web_flatten_dict, if find in_mesh_key use web_flatten_dict[in_mesh_key][geom_key], else return None
    
    Args:
        web_flatten_dict: Example: /aigc_cfs_3/layer_tex/mcwy/merge/web_0406/web_flatten.json
        in_mesh_key: obj key or None

    Returns:
        geom_png or None
    """
    try:
        if in_mesh_key is None or (in_mesh_key not in web_flatten_dict):
            print(f"[warn] invalid in_mesh_key {in_mesh_key}, return geom_png=None")
            return None

        query_value = web_flatten_dict[in_mesh_key][query_key]
        if query_key=="uv_pos":
            if os.path.exists(query_value):
                print(f"find valid {query_value } from in_mesh_key {in_mesh_key} and query_key {query_key}")
                return query_value
            else:
                print(f"[warn] have in_mesh_key {in_mesh_key} but can not find query_value={query_value} and query_key {query_key}")
                return None
        return query_value
    except Exception as e:
        print("error when query_value_from_key, keep None ", e)
        traceback.print_exc()
        return None


def Category_map_to_part_names(Category, replace_img_path):
    cname_part_name_map = {
        'hair': ["SM_Hair"],
        'trousers': ["SM_Bottom"],
        'outfit': ["SM_Outfit"],
        'top': ["SM_Top"],
        'shoe': ["SM_Shoe_Left", "SM_Shoe_Right"]
    }
    object_part_names = cname_part_name_map.get(Category, [])

    if not object_part_names or len(object_part_names) < 1:
        print(f"[error] invalid Category {Category}, cname_part_name_map=", cname_part_name_map)
        return False, [], []

    input_image_paths = [replace_img_path for name in object_part_names]

    flag = len(input_image_paths) > 0
    return flag, input_image_paths, object_part_names

def replace_mesh_glb(raw_glb, input_image_paths, object_part_names, out_glb,
                     blender_root="/usr/blender-3.6.2-linux-x64/blender", replace_py_name="replace_glb_part_uvtex.py"):
    replace_py = os.path.join(codedir, "replace_uv", replace_py_name)
    assert os.path.exists(replace_py), replace_py
    input_image_paths_str = "'{}' ".format("' '".join(input_image_paths))
    object_part_names_str = "'{}' ".format("' '".join(object_part_names))
    cmd = f"{blender_root} -b -P {replace_py} -- --source_mesh_path {raw_glb} --input_image_paths {input_image_paths_str} --object_part_names {object_part_names_str} --output_mesh_path {out_glb}"
    print(f'replace_mesh_glb cmd={cmd}')

    os.system(cmd)
    return os.path.exists(out_glb)


class TexGenConsumer:

    def __init__(
            self,
            cfg_json='tex_gen.json',
            model_name='uv_mcwy',
            device='cuda'):
        """tex_gen consumer(server) and backend producer.
        message producer(TexGenProducer in main_call_texgen.py) -> pulsar queue -> TexGenConsumer and send result to backend topic ->  BackendConsumer

        Args:
            cfg_json: relative name in codedir/configs. Defaults to 'tex_gen.json'.
            model_name: key of cfg, can be selected in webui. Defaults to 'uv_mcwy'.
            token: _description_. Defaults to 
            service_url: _description_. 
            generation_topic: _description_.
            backend_topic: _description_. 
            device: _description_. Defaults to 'cuda'.
        """
        ## 1. init cfg and model
        self.load_cfg(cfg_json, model_name=model_name)
        self.device = device
        self.log_dir = os.path.join(self.cfg_model.log_root_dir, time.strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(self.log_dir, exist_ok=True)

        cvt_ip, cvt_port = "localhost", "987"
        self.use_blender_server = is_grpc_server_running(cvt_ip, cvt_port)
        if self.use_blender_server:
            sys.path.append(os.path.join(codedir, "grpc_backend"))
            from grpc_backend.client_blendercvt import BlenderCVTClient
            self.blender_cvt_client = BlenderCVTClient(ip_port=f"{cvt_ip}:{cvt_port}")
        logging.info(f"use_blender_server: {self.use_blender_server}")

        logging.info(
            f'begin loading {self.model_name} model from {self.cfg_model.in_model_path}, Need a little time in T10')
        self.load_pipeline()
        logging.info(f"load_pipeline done")

        ## 2. init tex_gen consumer and backend producer
        tdmq_dict = self.cfg_model.tdmq
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),# 已授权角色密钥
            service_url=tdmq_dict.service_url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )

        self.consumer = self.client.subscribe(
            # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
            topic=tdmq_dict.generation_topic,
            # 订阅名称
            subscription_name="texgen-sub1",
            # 设置监听
            message_listener=self.on_message,
            # 设置订阅模式为 Shared（共享）模式
            consumer_type=ConsumerType.Shared,
            properties={"texture_generation": "TAGS"},
            # 配置从最早开始消费，否则可能会消费不到历史消息
            initial_position=InitialPosition.Earliest,
        )
        logging.info(f"init consumer done, topic={tdmq_dict.generation_topic}")

        self.producer_backend = self.client.create_producer(topic=tdmq_dict.backend_topic)
        logging.info(f"init producer_backend done, topic={tdmq_dict.backend_topic}")

        logging.info(f"begin listener")

        self.start()

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
        cfg_root = load_json(cfg_json_)  #
        assert len(cfg_root), f"invalid cfg_json: {cfg_json}"

        model_name_list = list(cfg_root.keys())
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

        try:
            if not os.path.exists(self.cfg_model.web_flatten_json):
                # when use relative path
                self.web_flatten_json = os.path.join(os.path.dirname(self.cfg_json), self.cfg_model.web_flatten_json)
            else:
                self.web_flatten_json = self.cfg_model.web_flatten_json   # when use abs path
            assert os.path.exists(self.web_flatten_json), f"can not find web_flatten_json: {self.web_flatten_json}"
            self.web_flatten_dict = load_json(self.web_flatten_json)
            logging.info(f"load web_flatten_json done from {self.web_flatten_json}, find {len(self.web_flatten_dict)} objs")

        except Exception as e:
            print("load web_flatten_json failed", e)
            raise ValueError("load web_flatten_json failed")

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
                self.cfg_model.pretrained_vae_model_name_or_path
                if "pretrained_vae_model_name_or_path" in self.cfg_model else None,
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
        else:
            raise NotImplementedError(f"invalid pipe_type {self.cfg_model.pipe_type}")
        return

    # ------------ interface functions ------------
    def on_message(self, consumer, msg):
        try:
            ## run core function
            job_id, success, result = self.parse_and_run(msg)
            feedback = "ok"
        except Exception as e:
            feedback = str(e)
            try:
                job_id = json.loads(msg.data())["parameter"]["job_id"]
            except Exception as e1:
                # 代码不应该运行到这里！ 无论任务怎么样，需要保证job_id传递的正确性，到这里肯定是因为发送格式错误
                job_id = "ERROR_texgen"
                logging.error(f"[ERROR] A serious interface error occurred. job_id could not be found.")
                print('[ERROR] msg', msg)

            job_id, success, result = job_id, False, f"error: {feedback}"
            print(f"Error occurred when query texture_generation: {str(e)}")
        finally:
            ## return result to backend topic
            out_dict = {
                "service_name": "texture_generation",
                "job_id": job_id,
                "flag": success,
                "result": result,
                "feedback": feedback,
                "receive_data": json.loads(msg.data())
            }
            out_data = json.dumps(out_dict)

            self.producer_backend.send_async(
                # 消息内容
                out_data.encode("utf-8"),
                # 异步回调
                callback=self.send_callback,
                # 消息参数
                properties={"texture_generation_results": "TAGS"},
                # 业务key
                # partition_key='key1'
            )
            self.consumer.acknowledge(msg)

    def send_callback(self, send_result, msg_id):
        print("[send-backend] send_result:{}  msg_id:{}".format(send_result, msg_id))

    def start(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()
            self.producer_backend.close()
            self.client.close()

    def wrap_output(self, out_mesh_paths_queryed):
        """convert ",".join([mesh]) from grpc to [mesh] list

        Args:
            out_mesh_paths_queryed: If '' is returned, the task failed. If run success, it is ",".join(out_mesh_paths)
        Return list of obj path
        """
        if not out_mesh_paths_queryed:
            return ['']
        out_mesh_paths = out_mesh_paths_queryed.split(',')

        return out_mesh_paths

    def parse_and_run(self, msg):
        """receive query data and run pipeline

        Args:
            msg: json.loads(msg.data()) = :
            {
            "service_name": "texture_generation",
            "parameter": {
                "job_id": job_id,
                "in_mesh_path": in_mesh_path,   # needed
                "in_mesh_key": in_mesh_key,   # optional
                "out_objs_dir": out_objs_dir, # needed
                "in_prompts": in_prompts, # optional
                "in_condi_img": in_condi_img, # optional
                "extra_param":{},
                "fast_mode_param" = {
                    "fast_mode": True,
                    "raw_glb": "xx/mesh.glb",
                    "out_glb": f"xx/replace_mesh_{job_id}.glb"
                } 
                "run_cfg": run_cfg  # optional, if None use default
            },
            }
               run_cfg = {
                "out_mesh_format": out_mesh_format,
                "uv_res": uv_res,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "ip_adapter_scale": ip_adapter_scale,
                "debug_save": debug_save,
            }
        Returns:
            job_id
            suc_flag=T/F
            results= list of obj path or [""]
        """
        # 1. parse data
        receive_data = json.loads(msg.data())
        service_name = receive_data.get("service_name", None)
        if service_name != "texture_generation":
            return False, f"invalid service_name: {service_name}"

        call_data = receive_data["parameter"]
        job_id = call_data["job_id"]
        logging.info(f"tex_gen consumer receive job_id:{job_id}")
        print('call_data:', call_data)

        func_name = call_data.get("func_name", "")
        if func_name == "infer_img":
            logging.info("run func_name=infer_img")
            suc_flag, results = self.infer_img(in_mesh_path=call_data.get("in_mesh_path", None),
                                               in_mesh_key=call_data.get("in_mesh_key", None),
                                               out_objs_dir=call_data.get("out_objs_dir", self.log_dir),
                                               in_prompts=call_data.get("in_prompts", ""),
                                               in_condi_img=call_data.get("in_condi_img", None),
                                               run_cfg=call_data.get("run_cfg", dict()))
            return job_id, suc_flag, results

        elif func_name == "batch_replace_glb":
            logging.info("run func_name=batch_replace_glb")
            return self.batch_replace_glb(job_id,
                                   mid_dict=call_data.get("mid_dict", {}),
                                   fast_mode_param=call_data.get("fast_mode_param", {}),
                                   extra_param=call_data.get("extra_param", {}))
        else:
            logging.info("run old pipe")

        
        ### TODO old pipe..
        run_cfg = call_data.get("run_cfg", dict())
        in_mesh_key = call_data.get("in_mesh_key", None)
        fast_mode_param = call_data.get("fast_mode_param", dict())

        out_objs_dir = call_data.get("out_objs_dir", self.log_dir)
        in_prompts = call_data.get("in_prompts", "")
        in_condi_img = call_data.get("in_condi_img", None)

        # TODO ["uv_pos"]
        in_geom_png = query_value_from_key(self.web_flatten_dict, in_mesh_key, query_key="uv_pos")
        if fast_mode_param.get("fast_mode", False) and in_geom_png is not None:
            # fast_mode
            suc_flag, results = False, [""]
            try:
                assert "raw_glb" in fast_mode_param, f"invalid fast_mode_param={fast_mode_param}"
                time_list = [('start', time.time())]
                in_prompts, in_condi_img_pils = self.pipeline.parse_in_prompts_and_in_condi_img(
                    in_prompts, in_condi_img)
                in_geom_pil = Image.open(in_geom_png)
                in_uv_geom_pils = [in_geom_pil] * len(in_prompts)
                time_list.append(('load_geom_done', time.time()))
                out_uv_pils = self.pipeline.infer_uv_xl_geom(
                    in_prompts,
                    in_uv_geom_pils,
                    in_condi_img_pils=in_condi_img_pils,
                    num_inference_steps=run_cfg.get("num_inference_steps", 20),
                    guidance_scale=run_cfg.get("guidance_scale", 9.0),
                    controlnet_conditioning_scale=run_cfg.get("controlnet_conditioning_scale", 0.8),
                    ip_adapter_scale=run_cfg.get("ip_adapter_scale", 0.8),
                )
                time_list.append(('infer_done', time.time()))
                # results = out_pngs, list of str
                suc_flag, out_pngs = self.pipeline.save_out_uv_pils(out_uv_pils,
                                                                   out_objs_dir,
                                                                   in_uv_geom_pils=in_uv_geom_pils,
                                                                   debug_save=run_cfg.get("debug_save", True))
                time_list.append(('save_done', time.time()))
                assert suc_flag, f"[ERROR] infer_uv_xl_geom failed job_id={job_id}"

                Category = query_value_from_key(self.web_flatten_dict, in_mesh_key, query_key="Category")
                suc_flag, input_image_paths, object_part_names = Category_map_to_part_names(Category, out_pngs[0])
                assert suc_flag, f"[ERROR] Category_map_to_part_names failed job_id={job_id}"

                raw_glb = fast_mode_param["raw_glb"]
                out_glb = fast_mode_param.get("out_glb", os.path.join(out_objs_dir, f"replace_mesh_{job_id}.glb"))

                logging.info(f"tex_gen begin replace_mesh_glb job_id:{job_id}")
                suc_flag = replace_mesh_glb(raw_glb, input_image_paths, object_part_names, out_glb)
                if suc_flag:
                    results = [out_glb]

                time_list.append(('replacemesh_done', time.time()))
                log_time_list(time_list)
            except Exception as e:
                logging.error(f"ERROR: obj_teximguv failed {e}")
                traceback.print_exc()

            logging.info(f"tex_gen fast_mode done job_id:{job_id}, suc_flag={suc_flag}, results:{results}")

        else:
            # 2.1 original mode. in/out mesh
            in_mesh_path = query_mesh_from_key(self.web_flatten_dict, call_data["in_mesh_path"], in_mesh_key)
            preprocess_param = call_data.get("extra_param", dict()).get("preprocess", None)
            if preprocess_param == "add_image":
                mid_obj = os.path.join(call_data["out_objs_dir"], "preprocess.obj")
                if blender_add_image(in_mesh_path, mid_obj):
                    in_mesh_path = mid_obj
                    print('preprocess-add_image ok')

            results = self.run_tex_gen(
                job_id=job_id,
                pipe_type=call_data.get("pipe_type", "tex_uv"),
                in_mesh_path=in_mesh_path,
                out_objs_dir=call_data["out_objs_dir"],
                in_prompts=in_prompts,
                in_condi_img=in_condi_img,
                in_geom_png=in_geom_png,
                out_mesh_format=run_cfg.get("out_mesh_format", "glb"),
                uv_res=run_cfg.get("uv_res", 1024),
                num_inference_steps=run_cfg.get("num_inference_steps", 20),
                guidance_scale=run_cfg.get("guidance_scale", 9.0),
                controlnet_conditioning_scale=run_cfg.get("controlnet_conditioning_scale", 0.8),
                ip_adapter_scale=run_cfg.get("ip_adapter_scale", 0.8),
                debug_save=run_cfg.get("debug_save", True),
            )
            suc_flag = True if results else False
            results = self.wrap_output(results)

        logging.info(f"tex_gen consumer run done job_id:{job_id}, results:{results}")
        return job_id, suc_flag, results

    def infer_img(self,
                  in_mesh_path=None,
                  in_mesh_key=None,
                  out_objs_dir=None,
                  in_prompts="",
                  in_condi_img="",
                  run_cfg={}):

        suc_flag, out_pngs = False, [""]
        try:
            in_geom_png = query_value_from_key(self.web_flatten_dict, in_mesh_key, query_key="uv_pos")
            time_list = [('start', time.time())]
            if in_geom_png is None:
                out_pngs = ["only support mesh key mode now."]
                raise NotImplementedError("only support mesh key mode now.")

            in_prompts, in_condi_img_pils = self.pipeline.parse_in_prompts_and_in_condi_img(
                in_prompts, in_condi_img)
            in_geom_pil = Image.open(in_geom_png)
            in_uv_geom_pils = [in_geom_pil] * len(in_prompts)
            time_list.append(('load_geom_done', time.time()))
            out_uv_pils = self.pipeline.infer_uv_xl_geom(
                in_prompts,
                in_uv_geom_pils,
                in_condi_img_pils=in_condi_img_pils,
                num_inference_steps=run_cfg.get("num_inference_steps", 20),
                guidance_scale=run_cfg.get("guidance_scale", 9.0),
                controlnet_conditioning_scale=run_cfg.get("controlnet_conditioning_scale", 0.8),
                ip_adapter_scale=run_cfg.get("ip_adapter_scale", 0.8),
            )
            time_list.append(('infer_done', time.time()))
            # results = out_pngs, list of str
            suc_flag, out_pngs = self.pipeline.save_out_uv_pils(out_uv_pils,
                                                                out_objs_dir,
                                                                in_uv_geom_pils=in_uv_geom_pils,
                                                                debug_save=run_cfg.get("debug_save", True))
            time_list.append(('save_done', time.time()))
            log_time_list(time_list)
            return suc_flag, out_pngs

        except Exception as e:
            logging.error(f"ERROR: infer_img failed {e}")
            traceback.print_exc()

        return suc_flag, out_pngs

    def batch_replace_glb(self, job_id, mid_dict={}, fast_mode_param={}, extra_param={}):
        """for call_batch_texreplace.step2

        Args:
            mid_dict:  once_job_id: {"success_flag": success_flag, "result_pngs": result_pngs, "meta": meta}
            fast_mode_param: _description_. Defaults to {}.
            extra_param: _description_. Defaults to {}.
        """
        suc_flag, results = False, [""]
        try:
            input_image_paths_all, object_part_names_all = [], []
            if not mid_dict:
                return job_id, suc_flag, results
            for once_job_id, one_part_dict in mid_dict.items():
                if not one_part_dict.get("success_flag", False):
                    logging.error(f"not suc for {once_job_id} ")
                    continue
                result_pngs = one_part_dict.get("result_pngs", [])
                if not result_pngs or len(result_pngs) < 1:
                    logging.error(f"[ERROR] suc but invalid result_pngs??? for {once_job_id} ")
                    continue

                meta = one_part_dict.get("meta", {})
                assert "in_mesh_key" in meta, f"invalid meta format {meta}"
                Category = query_value_from_key(self.web_flatten_dict, meta["in_mesh_key"], query_key="Category")
                suc_flag, input_image_paths, object_part_names = Category_map_to_part_names(Category, result_pngs[0])
                assert suc_flag, f"[ERROR] Category_map_to_part_names failed job_id={job_id}, once_job_id={once_job_id}"

                input_image_paths_all += input_image_paths
                object_part_names_all += object_part_names

            raw_glb = fast_mode_param["raw_glb"]
            out_glb = fast_mode_param.get("out_glb", os.path.join(os.path.dirname(fast_mode_param["raw_glb"]), f"replace_mesh_{job_id}.glb"))

            logging.info(f"tex_gen begin replace_mesh_glb job_id:{job_id}")
            suc_flag = replace_mesh_glb(raw_glb, input_image_paths_all, object_part_names_all, out_glb)
            results = [out_glb] if suc_flag else [f"replace_mesh_glb failed for {job_id}"]

        except Exception as e:
            logging.error(f"[ERROR]: obj_teximguv failed {e}")
            traceback.print_exc()
        return job_id, suc_flag, results

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
            output_obj_path = os.path.join(os.path.dirname(in_mesh_path), filename, "cvt_obj/mesh.obj")
            if self.use_blender_server:
                flag = self.blender_cvt_client.interface_glb_to_obj(in_mesh_path, output_obj_path)
            else:
                flag = cvt_glb_to_obj(in_mesh_path, output_obj_path)
            if not flag:
                raise ValueError(f"cvt glb to obj failed! use_blender_server:{self.use_blender_server}")
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

    ## ------------ core functions to run each pipeline ---------------
    def run_tex_gen(
        self,
        job_id,
        pipe_type,
        in_mesh_path,
        out_objs_dir,
        in_prompts=None,
        in_condi_img=None,
        in_geom_png=None,
        in_obj_type=None,
        out_mesh_format="glb",
        uv_res=1024,
        num_inference_steps=20,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        debug_save=True,
    ):
        """[with obj] feed text and image, get uv texture from mesh, results is out_mesh_paths(list of {out_mesh_format} path) or ""

        Args:
            job_id: uuid from request
            pipe_type: tex_control or tex_imguv
            in_mesh_path: raw mesh path with uv coord, can be obj or glb
            out_objs_dir: output dir with multi output objs
            in_prompts: input text, list of text or None
            in_condi_img: condi img path or None
            in_geom_png: input pre-rendered geom png (masked uv pos)
            in_obj_type: obj type(top, dress, and so on) or None
            out_mesh_format: output mesh as glb or obj
            uv_res: generate uv resolution
            num_inference_steps: diffusion loop steps
            guidance_scale: text guidance scale:The larger the value, the closer it is to the text prompt. Defaults to 9.0.
            controlnet_conditioning_scale (int, optional): The higher the value, the better the control but the worse the quality. in [0, 1]
            ip_adapter_scale (int, optional): The larger the value, the closer it is to the image. in [0, 1]
            debug_save: save debug result if True.
        Return:
            results(string): ",".join(list of out_obj_paths) if succeed, else ""
        Raises:
            e: _description_
        """
        log_name = f"run_{pipe_type}"
        logging.info(
            f'{log_name} input in_mesh_path=\"{in_mesh_path}\", in_prompts={in_prompts} in_condi_img= {in_condi_img}, in_geom_png={in_geom_png}'
        )
        logging.info(f'job_id: {job_id}')
        results = ""
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
                    in_geom_png=in_geom_png,
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
            else:
                raise ValueError(f"invalid pipe_type {pipe_type}")

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
        except Exception as e:
            logging.error(f'{log_name} failed, job_id={job_id}')
            raise ValueError(e)

        logging.info(
            f'{log_name}  done, in_mesh_path="{in_mesh_path}", in_prompts={in_prompts} in_condi_img= {in_condi_img}", results={results}'
        )

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tqmd pulsar main consumer')
    parser.add_argument('--cfg_json', type=str, default='tex_gen.json')
    parser.add_argument('--model_name',
                        type=str,
                        default='uv_mcwy',
                        help='select model. can be uv_mcwy, control_mcwy, imguv_mcwy, imguv_lowpoly, pipe_type_dataset')
    args = parser.parse_args()

    consumer = TexGenConsumer(args.cfg_json, args.model_name)
