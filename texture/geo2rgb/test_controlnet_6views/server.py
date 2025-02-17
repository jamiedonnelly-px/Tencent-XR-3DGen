import time
import argparse
import logging
import pulsar
from _pulsar import ConsumerType, InitialPosition, LoggerLevel
from easydict import EasyDict as edict
import json
import os
import sys
import torch

import numpy as np
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler
from pipeline.pipeline_zero123plus import Zero123PlusPipeline
from sam_preprocess.run_sam import process_image, process_image_path
from sam_preprocess.utils import remove_backgroud_whitebg

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.getLogger("pulsar").setLevel(logging.WARNING)

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)


class Zero123plusServer:
    def __init__(
            self,
            cfg_json='config.json',
            model_name="zero123plus_v10",
            result_rmbg=True):
        """
        Args:
            cfg_json: service config json path
        """
        ## 1. init cfg and model
        self.load_cfg(cfg_json, model_name)
        self.result_rmbg=result_rmbg
        self.device = torch.device("cuda:0")
        self.log_dir = os.path.join(self.model_config.log_dir, time.strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(self.log_dir, exist_ok=True)

        logging.info(
            f'begin loading {self.model_name} model from {self.model_config.model_ckpt_dir}')
        self.load_pipeline(torch_dtype=torch.float16)
        logging.info(f"load_pipeline done")

        ## 2. init tex_gen consumer and backend producer
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(self.tdmq_config.token),# 已授权角色密钥
            service_url=self.tdmq_config.url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )

        self.producer_backend = self.client.create_producer(topic=self.tdmq_config.backend_topic)
        logging.info(f"init producer_backend done, topic={self.tdmq_config.backend_topic}, begin listener")

        self.consumer = self.client.subscribe(
            # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
            topic=self.tdmq_config.topic,
            # 订阅名称
            subscription_name="zero123plus",
            # 设置监听
            message_listener=self.on_message,
            # 设置订阅模式为 Shared（共享）模式
            consumer_type=ConsumerType.Shared,
            properties={"zero123plus": "TAGS"},
            # 配置从最早开始消费，否则可能会消费不到历史消息
            initial_position=InitialPosition.Earliest,
        )
        logging.info(f"init consumer done, topic={self.tdmq_config.topic}")

        self.start()

    def load_cfg(self, cfg_json, model_name="zero123plus_v10"):
        """
        Args:
            cfg_json: relative path in codedir/configs
        """
        logging.info(f"input of load_cfg: cfg_json: {cfg_json}")
        self.configs_dir = os.path.join(codedir, 'configs')
        cfg_json_ = os.path.join(self.configs_dir, cfg_json)
        if not os.path.exists(cfg_json_):
            logging.error(f"config path not exists: {cfg_json_}")
        assert os.path.exists(cfg_json_), f"can not find valid cfg_json: {cfg_json}"

        with open(cfg_json_, 'r') as fr:
            json_dict = edict(json.load(fr))

        self.model_config = json_dict.model_config
        self.model_config.model_name = model_name
        self.model_config.model_ckpt_dir = os.path.join(self.model_config.model_ckpt_dir, model_name)
        self.model_config.log_dir = os.path.join(self.model_config.model_ckpt_dir, "log")

        self.model_name = self.model_config.model_name
        self.model_ckpt_dir = self.model_config.model_ckpt_dir
        self.tdmq_config = json_dict.tdmq_config
        logging.info(f"load_cfg done with self.model_name:{self.model_name}")
        return

    def load_pipeline(self, torch_dtype=torch.float32):
        self.zero123plus_pipeline = Zero123PlusPipeline.from_pretrained(
                            self.model_ckpt_dir,
                            torch_dtype=torch_dtype,
                            local_files_only=True
                        )
        # Feel free to tune the scheduler
        self.zero123plus_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.zero123plus_pipeline.scheduler.config,
            timestep_spacing='trailing'
        )
        self.zero123plus_pipeline.to(self.device)
        return

    # ------------ interface functions ------------
    def on_message(self, consumer, msg):
        try:
            logging.info(f"\n===============\nreceiving message: {msg}\n")
            ## run core function
            job_id, success, result = self.parse_and_run(msg)
            feedback = "ok"
        except Exception as e:
            feedback = str(e)
            job_id, success, result = "", False, ""
            print(f"Error occurred when query zero123plus: {str(e)}")
        finally:
            ## return result to backend topic
            out_dict = {
                "service_name": "zero123plus",
                "job_id": job_id,
                "flag": success,
                "result": result,
                "feedback": feedback
            }
            logging.info(f"\nfeedback message: {out_dict}\n===============\n")
            out_data = json.dumps(out_dict)

            self.producer_backend.send(out_data.encode("utf-8"))
            self.consumer.acknowledge(msg)

    def start(self):
        logging.info("server started, listening...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()
            self.producer_backend.close()
            self.client.close()

    def inference(self, image_origin_path, image_path, wh_ratio=0.8, cfg_scale=3.0, step_num=75, job_id_dir="", **kwargs):
        if not os.path.exists(image_origin_path):
            logging.error(f"file not exists: {image_origin_path}")
            return None
        try:
            image = process_image_path(image_origin_path, bg_color=255, wh_ratio=wh_ratio)

            ### todo save processed image
            image.save(image_path)

            image = Image.fromarray(np.array(image).astype(np.uint8)[..., :3]).resize((512, 512))

            mv_imgs = self.zero123plus_pipeline(image, num_inference_steps=step_num, guidance_scale=cfg_scale)
            return mv_imgs
        except:
            logging.error(f"inference failed!")
            return None

    def parse_and_run(self, msg):
        """receive query data and run pipeline

        Args:
            msg: json.loads(msg.data()) = :
            {
            "service_name": "zero123plus",
            "parameter": {
                "job_id": job_id,
                "in_image_path": in_image_path,   # needed
                "out_save_dir": out_save_dir, # needed
                "wh_ratio": 0.8, # optional
                "cfg_scale": 3.0, # optional
                "step_num": 75, # optional
            },
        Returns:
            job_id
            suc_flag=T/F
            results= result dict:
            {   
                "job_id": job_id,
                "image_origin_path": call_data["in_image_path"],
                "image_path": image_preprocessed_path,
                "wh_ratio": call_data.get("wh_ratio", 0.8),
                "cfg_scale": call_data.get("cfg_scale", 3.0),
                "step_num": call_data.get("step_num", 75),
                "result_rmbg": self.result_rmbg,
                "image_npy_path": save_path,
                "normal_npy_path":  None, # optional
                "job_id_dir": out_npy_dir
            }
        """
        # 1. parse data
        receive_data = json.loads(msg.data())
        service_name = receive_data.get("service_name", None)
        if service_name != "zero123plus":
            return False, f"invalid service_name: {service_name}"

        call_data = receive_data["parameter"]

        job_id = call_data["job_id"]
        logging.info(f"zero123plus server receive job_id:{job_id}")
        out_save_dir = call_data.get("out_save_dir", None)
        assert out_save_dir is not None, "message passed in must have out_save_dir defined"
        job_id_dir = os.path.join(out_save_dir, f"{job_id}")
        os.makedirs(job_id_dir, exist_ok=True)

        filename = os.path.splitext(os.path.basename(call_data["in_image_path"]))[0]
        save_path = os.path.join(job_id_dir, filename + ".npy")
        image_preprocessed_path = os.path.join(job_id_dir, f"preprocess_z123.jpg")
        image_save_path = os.path.join(job_id_dir, f"result_z123.jpg")

        inference_param = {
                            "job_id": job_id,
                            "image_origin_path": call_data["in_image_path"],
                            "image_path": image_preprocessed_path,
                            "wh_ratio": call_data.get("wh_ratio", 0.8),
                            "cfg_scale": call_data.get("cfg_scale", 3.0),
                            "step_num": call_data.get("step_num", 75),
                            "result_rmbg": self.result_rmbg,
                            "image_npy_path": save_path,
                            "normal_npy_path": None,
                            "job_id_dir": job_id_dir
                           }
        
        logging.info(f"===inference info===: \n {inference_param} \n ====== \n")

        results = self.inference(**inference_param)

        suc_flag = True if results is not None else False

        if suc_flag:
            mv_imgs = results.images[0]
            mv_imgs.save(image_save_path)
            mv_imgs = np.array(mv_imgs).transpose(2, 0, 1)
            ## zero123plus_v4.4.1
            mv_imgs = np.stack([mv_imgs[:, :256, :256], mv_imgs[:, 256:512, :256],
                                mv_imgs[:, 512:768, :256], mv_imgs[:, 768:, :256],
                                mv_imgs[:, :256, 256:], mv_imgs[:, 256:512, 256:],
                                mv_imgs[:, 512:768, 256:], mv_imgs[:, 768:, 256:],
                                ], axis=0)

            # ### zero123plus_v4.7
            # mv_imgs = np.stack([mv_imgs[:, :256, :256], mv_imgs[:, :256, 256:],
            #                     mv_imgs[:, 256:512, :256], mv_imgs[:, 256:512, 256:],
            #                     mv_imgs[:, 512:768, :256], mv_imgs[:, 512:768, 256:],
            #                      mv_imgs[:, 768:, :256], mv_imgs[:, 768:, 256:]
            #                     ], axis=0)  ### [8, 3, 256, 256]
            if self.result_rmbg:
                try:
                    mv_imgs = remove_backgroud_whitebg(mv_imgs)
                except:
                    suc_flag = False
                    logging.error("remove generated bg failed!")
        else:
            inference_param["image_npy_path"] = ""
            return job_id, suc_flag, inference_param

        try:
            np.save(save_path, mv_imgs.astype(np.uint8))
        except:
            suc_flag = False
            logging.warning("npy save failed, maybe disk is full!")

        logging.info(f"zero123plus server run done job_id:{job_id}, results:{save_path}")
        return job_id, suc_flag, inference_param



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tqmd pulsar server consumer')
    parser.add_argument('--cfg_json', type=str, default='zero123plus_stable.json')
    parser.add_argument("--model_name", type=str, default="zero123plus_v10")
    parser.add_argument("--result_rmbg", type=str, default="true") ## "true" | "false" 
    args = parser.parse_args()

    if args.result_rmbg == "false":
        args.result_rmbg = False
    else:
        args.result_rmbg = True
    consumer = Zero123plusServer(args.cfg_json, args.model_name, args.result_rmbg)