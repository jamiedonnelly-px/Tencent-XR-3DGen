import pulsar
from _pulsar import ConsumerType, LoggerLevel
import uuid
import json
import logging
import time
import argparse
import os
import threading
from easydict import EasyDict as edict

import redis

import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


### utils func
def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)
    return data


def init_job_id():
    return str(uuid.uuid4())


def find_cfg_path(cfg_json):
    if os.path.exists(cfg_json):
        cfg_json_ = os.path.abspath(cfg_json)
    else:
        configs_dir = os.path.join(codedir, 'configs')
        cfg_json_ = os.path.join(configs_dir, cfg_json)
    assert os.path.exists(cfg_json_), f"can not find valid cfg_json: {cfg_json}, {cfg_json_}"
    cfg_dict = load_json(cfg_json_)
    return cfg_json_, cfg_dict

class TexGenProducer:

    def __init__(
        self,
        client_cfg_json="client_texgen.json",
        model_name="uv_mcwy",
        device='cuda',
    ):
        self.load_run_cfg(client_cfg_json, model_name=model_name)
        self.device = device
        logging.info(f"load run cfg done from {client_cfg_json}")

        tdmq_dict = self.run_cfg.tdmq
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),# 已授权角色密钥
            service_url=tdmq_dict.service_url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )
        self.producer = self.client.create_producer(topic=tdmq_dict.generation_topic)
        logging.info(f"init texgen producer done, topic={tdmq_dict.generation_topic}")

    ### external interface functions
    def interface_query_text(
        self,
        job_id,
        in_mesh_path,
        in_prompts,
        in_mesh_key=None,
        out_objs_dir=None,
        in_obj_type="",
        pipe_type="tex_uv",
    ):
        """query text only mode

        Args:
            job_id(string): uuid
            in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
            in_prompts(string/ list of string): 文本提示,可以是字符串或字符串list
            in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
            out_objs_dir(string): 输出文件夹路径
            in_obj_type(string): 可以就用默认值, obj_type or '' TODO(csz, not used now)
            pipe_type(string): 可以就用默认值, need be tex_uv, tex_control or tex_imguv, select pipeline
        Returns:
            send_flag: send succeed or failed
        """
        logging.info(f"begin interface_query_text of job_id:{job_id}")
        send_flag = self.call_obj_texgen(
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

        return send_flag

    def interface_query_image(
        self,
        job_id,
        in_mesh_path,
        in_condi_img,
        in_mesh_key=None,
        out_objs_dir=None,
        in_obj_type="",
        pipe_type="tex_uv",
    ):
        """query image only mode

        Args:
            job_id(string): uuid
            in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
            in_condi_img(string): 控制图片路径. condi img path
            in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
            out_objs_dir(string): 输出文件夹路径
            in_obj_type(string): 可以就用默认值, obj_type or '' TODO(csz, not used now)
            pipe_type(string): 可以就用默认值, need be tex_uv, tex_control or tex_imguv, select pipeline
        Returns:
            send_flag: send succeed or failed
        """
        logging.info(f"begin interface_query_image of job_id:{job_id}")
        send_flag = self.call_obj_texgen(
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

        return send_flag

    def interface_query_text_image(
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
        """query text+image mode

        Args:
            job_id(string): uuid
            in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
            in_prompts(string/ list of string): 文本提示,可以是字符串或字符串list
            in_condi_img(string): 控制图片路径. condi img path
            in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
            out_objs_dir(string): 输出文件夹路径
            in_obj_type(string): 可以就用默认值, obj_type or '' TODO(csz, not used now)
            pipe_type(string): 可以就用默认值, need be tex_uv, tex_control or tex_imguv, select pipeline
        Returns:
            send_flag: send succeed or failed
        """
        mix_ip_division_scale = 4.0 if "mix_ip_division_scale" not in self.run_cfg else self.run_cfg.mix_ip_division_scale
        send_flag = self.call_obj_texgen(
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
            ip_adapter_scale=self.run_cfg.ip_adapter_scale / mix_ip_division_scale,      # use small ip scale when mix mode
        )
        return send_flag

    ###### Ignore the code that follows
    ### pulsar functions
    def send_messages(self, send_data: dict):
        self.producer.send_async(
            # 消息内容
            send_data.encode("utf-8"),
            # 异步回调
            callback=self.send_callback,
            # 消息参数
            properties={"texture_generation": "TAGS"},
            # 业务key
            # partition_key='key1'
        )

    def send_callback(self, send_result, msg_id):
        print("[send-tex] send_result:{}  msg_id:{}".format(send_result, msg_id))

    def make_message(
            self,
            job_id="",
            in_mesh_path=None,
            in_mesh_key=None,
            out_objs_dir=None,
            in_prompts="",
            in_condi_img=None,
            run_cfg=dict(),
    ):
        send_dict = {
            "service_name": "texture_generation",
            "parameter": {
                "job_id": job_id,
                "in_mesh_path": in_mesh_path,
                "in_mesh_key": in_mesh_key,
                "out_objs_dir": out_objs_dir,
                "in_prompts": in_prompts,
                "in_condi_img": in_condi_img,
                "run_cfg": run_cfg
            },
        }
        send_data = json.dumps(send_dict)
        return send_data

    def close(self):
        logging.info("closing....")
        time.sleep(3)
        self.producer.close()
        self.client.close()
        logging.info("closed")

    ### core function
    def call_obj_texgen(
        self,
        job_id,
        pipe_type="tex_uv",
        in_mesh_path=None,
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
        """send message, call texgen

        Args:
            job_id(string): uuid
            pipe_type(string): need be tex_uv(sdxl), tex_control(sd) or tex_imguv, select pipeline
            in_mesh_path(string): raw obj/glb path with uv coord, can be None when get valid in_mesh_key
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
        try:
            if out_objs_dir is None:
                log_root_dir = self.run_cfg.log_root_dir
                timestamp = int(time.time())
                unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
                out_objs_dir = os.path.join(log_root_dir, str(unique_id))
                os.makedirs(out_objs_dir, exist_ok=True)

            # 
            logging.info(
                f"begin client_obj_tex_gen, job_id {job_id} with pipe_type {pipe_type} and in_mesh_key {in_mesh_key}, in_mesh_path {in_mesh_path}"
            )

            ### begin
            run_cfg = {
                "out_mesh_format": out_mesh_format,
                "uv_res": uv_res,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "ip_adapter_scale": ip_adapter_scale,
                "debug_save": debug_save,
            }
            send_data = self.make_message(
                job_id=job_id,
                in_mesh_path=in_mesh_path,
                in_mesh_key=in_mesh_key,
                out_objs_dir=out_objs_dir,
                in_prompts=in_prompts,
                in_condi_img=in_condi_img,
                run_cfg=run_cfg,
            )
            self.send_messages(send_data)

            logging.debug(str(send_data))
        except Exception as e:
            print(f"run call_obj_texgen failed! {e}")
            return False
        return True

    ##### Internal functions
    def load_run_cfg(self, cfg_json, model_name=None):
        """set self.model_name_list and self.run_cfg from json

        Args:
            cfg_json: abs path or relative name in codedir/configs
            model_name: model key. Defaults to None.
        """
        logging.info(f"input of load_run_cfg: cfg_json: {cfg_json}, model_name:{model_name}")
        cfg_json_, cfg_root = find_cfg_path(cfg_json)
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
            logging.warn(f"invalid model_name in cfg_json: {cfg_json}, model_name:{model_name}, juet select first")
            model_name = model_name_list[0]
            run_cfg = cfg_root[model_name]

        self.model_name_list = model_name_list
        self.run_cfg = edict(run_cfg)
        self.cfg_json = cfg_json_
        self.model_name = model_name

        logging.info(f"load_run_cfg done with self.model_name:{self.model_name} with input model_name {model_name}")
        return self.run_cfg


class BackendConsumer:

    def __init__(
        self,
        client_cfg_json="client_texgen.json",
        model_name="uv_mcwy",
    ):
        _, cfg_root = find_cfg_path(client_cfg_json)
        assert model_name in cfg_root, f"can not find {model_name} in {client_cfg_json}"
        tdmq_dict = edict(cfg_root[model_name]).tdmq

        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),  # 已授权角色密钥
            service_url=tdmq_dict.service_url,  # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn))
        self.consumer = self.client.subscribe(
            # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
            topic=tdmq_dict.backend_topic,
            # 订阅名称
            subscription_name='backend-tex-sub',
            # 设置监听
            message_listener=self.on_message,
            # 设置订阅模式为 Shared（共享）模式
            consumer_type=ConsumerType.Shared,
            consumer_name="结果消费者",
            properties={'tag2': 'TAGS'},
        )

        self.redis_db = redis.StrictRedis(host=tdmq_dict.redis_host,
                                          port=tdmq_dict.redis_port,
                                          password=tdmq_dict.redis_password)
        self.redis_db.set("debug", json.dumps({
            "service_name": "hello",
            "success": True,
        }))
        data = json.loads(self.redis_db.get("debug"))
        print('test db ', data)
        self.stop_flag = False
        logging.info(f"init BackendConsumer done from {client_cfg_json} and {model_name}")


    def parse_and_run(self, msg):
        """receive result of texture_generation

        Args:
            msg: json.loads(msg.data()) = :
            out_dict = {
                "service_name": "texture_generation",
                "job_id": job_id,   # str uuid
                "flag": success,    # T/F
                "result": result,   # 输出的结果list of obj path or [""]
                "feedback": feedback # 报错信息(如果有)
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
            return receive_data.get("job_id", "-1"), False, f"invalid service_name: {service_name}!=texture_generation"

        job_id = receive_data["job_id"]
        success, result = receive_data["flag"], receive_data["result"]
        feedback = receive_data["feedback"]
        ## log error
        if not success:
            logging.error(f"error reason: {feedback}")
        return job_id, success, result

    def on_message(self, consumer, msg):
        """backend异步用法: 接受到算法结果的消息时的回调函数, 消费的同时会往redis写入缓存结果, 供轮循用

        Args:
            consumer: _description_
            msg: _description_
        """
        try:
            ## run core function
            job_id, success, result = self.parse_and_run(msg)
        except Exception as e:
            job_id, success, result = "", False, ""
            print(f"Error occurred when receive result of texture_generation: {str(e)}")
        finally:
            logging.info(f"[receive] result of texture_generation, job_id:{job_id}, success:{success}, result:{result}")
            # 无论是哪个后端消费者消费, 都往redis里写数据, 保证数据不丢；然后后端生产者可以根据job id去查询结果
            self.redis_db.set(job_id,
                              json.dumps({
                                  "service_name": "texture_generation",
                                  "success": success,
                                  "result": result
                              }))
            self.consumer.acknowledge(msg)


    def start(self):
        try:
            while self.stop_flag == False:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()
            self.client.close()

    def start_consumer(self):
        self.start()
    
    def close(self):
        self.stop_flag = True
        
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

    ### run backend consumer, receive result of tex_gen
    backend_consumer = BackendConsumer(args.client_cfg_json, args.model_name)
    consumer_thread = threading.Thread(target=backend_consumer.start_consumer)
    consumer_thread.start()
    time.sleep(1)
    logging.info("run backend consumer done")

    ### prepare example data
    model_name = args.model_name
    if model_name == "uv_mcwy" or model_name == 'control_mcwy':
        # in_mesh_path = f"/aigc_cfs_3/layer_tex/mcwy_2/2024/MCWY_2_Bottom/BR_BTM_5/uv_condition/mesh.obj"
        in_mesh_path = f"/aigc_cfs_gdp/Asset/designcenter/clothes/convert/mcwy2/remove_skin_mesh/meshes/Top/BR_TOP_1_F_T/BR_TOP_1_fbx2020.obj"
        in_mesh_key = "BR_TOP_1_F_T"
        # out_objs_dir = f"/aigc_cfs_gdp/cfs1/sz/server/tex_gen/client_log/mcwy_debug_gdp_{model_name}"
        out_objs_dir = f"/aigc_cfs_3/sz/server/tex_gen/pulsar_log/mcwy_debug_{model_name}"
        # out_objs_dir = f"/aigc_cfs_gdp/cfs1/sz/server/tex_gen/client_log/mcwy_debug_gdp_{args.model_name}"
        in_condi_img = "/aigc_cfs/Asset/designcenter/clothes/render_part2/render_data/dress/render_data/DR_640_F_A/Dresses_F_A_DR_640_F_A_DR_640_fbx2020_output_512_MightyWSB/color/cam-0032.png"
        in_prompts = "indian style"
    else:
        print('invalid model_key')
        exit()

    # 1. init producer
    producer = TexGenProducer(args.client_cfg_json, args.model_name)
    query_job_ids = set()

    # 2. run tex_gen producer, call query
    # *As an example of an external call*
    ### text only
    job_id = init_job_id()
    query_job_ids.add(job_id)
    producer.interface_query_text(
        job_id,
        in_mesh_path,
        in_prompts,
        in_mesh_key=in_mesh_key,
        out_objs_dir=os.path.join(out_objs_dir, "query_key"),
    )
    time.sleep(0.2)

    ### image only
    job_id = init_job_id()
    query_job_ids.add(job_id)
    producer.interface_query_image(job_id,
                                   in_mesh_path,
                                   in_condi_img,
                                   in_mesh_key=in_mesh_key,
                                   out_objs_dir=os.path.join(out_objs_dir, f"query_image"))
    time.sleep(0.2)

    ### text + image
    job_id = init_job_id()
    query_job_ids.add(job_id)
    producer.interface_query_text_image(job_id,
                                        in_mesh_path,
                                        in_prompts,
                                        in_condi_img,
                                        in_mesh_key=in_mesh_key,
                                        out_objs_dir=os.path.join(out_objs_dir, f"query_mix"))
    time.sleep(0.2)

    # # (batch request)
    # for i in range(10):
    #     job_id = init_job_id()
    #     query_job_ids.add(job_id)
    #     producer.interface_query_text(
    #         job_id,
    #         in_mesh_path,
    #         in_prompts,
    #         in_mesh_key=in_mesh_key,
    #         out_objs_dir=os.path.join(out_objs_dir, f"query_key_{i}"),
    #     )

    # 3. query results. backend同步用法: 从数据库里轮循结果, 后端形成一个闭环.
    print('query_job_ids ', query_job_ids)
    start_time = time.time()
    timeout = 30 * len(query_job_ids)
    while query_job_ids:
        for job_id in list(query_job_ids):
            job_data = backend_consumer.redis_db.get(job_id)
            if job_data:
                print(json.loads(job_data))
                print(f"[[Received]] job_data for job_id {job_id}: {job_data.decode('utf-8')}")
                query_job_ids.remove(job_id)
                print('the reset of query_job_ids ', query_job_ids)
        time.sleep(0.1)
        if time.time() - start_time >= timeout:
            logging.error("Timeout: Exiting the loop.")
            break

    # 4. close producer
    producer.close()

    logging.info("test done, you can shut done with ctrl+c")

    consumer_thread.join()
