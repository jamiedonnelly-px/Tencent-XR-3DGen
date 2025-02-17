import sys
import os
import redis
# sys.path.append(os.path.abspath("/aigc_cfs_2/zacheng/MMD_NPU_dmmd/"))
# sys.path.append(os.path.abspath("/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus"))
sys.path.append(os.path.abspath("/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus_gray"))
# from src.diffusers import MMDiffusionDepthToRgbImagePipeline as MMDiffusionDepthToRgbImagePipeline
# from src.diffusers import MMDiffusionNormalToRgbImagePipeline_dino_vit_4views_cuda as MMDiffusionDepthToRgbImagePipeline
from server_orths_3views.inference_orths_class import run_xyz2rgb
# from inference import run_d2rgb
import torch
import argparse
import pulsar
from _pulsar import ConsumerType, InitialPosition, LoggerLevel
import json
import socket
import time
import traceback
from pdb import set_trace as st

from transformers import ViTImageProcessor, ViTModel
from transformers import ViTFeatureExtractor

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
# from segment_anything import sam_model_registry, SamPredictor
from utils_use.utils_seg_rmbg import RMBG, repadding_rgba_image

class Service:
    
    def __init__(self, cfg_json):
        
        with open(cfg_json) as json_file:
            cfg = json.load(json_file)
            
        
        self.service_name = cfg['service_name']
        # self.in_pose_npy = cfg['data']['pose_npy']

        self.controlnet = ControlNetModel.from_pretrained(
            # "/aigc_cfs/xibinsong/models/3view_models/controlnet-8000/controlnet", torch_dtype=torch.float16
            "/aigc_cfs_gdp/xibin/z123_control/models/3view_models/controlnet-8000/controlnet", torch_dtype=torch.float16
            )

        device = torch.device(f"cuda:{'0'}" if torch.cuda.is_available() else "cpu")
        self.rmbg = RMBG(device)

        print("begin topic message !")
            
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(cfg['tdmq']['token']),
            service_url=cfg['tdmq']['url'],
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
        )
        
        self.producer = self.client.create_producer(topic=cfg['tdmq']['to_backend_topic'])       
        
        self.consumer = self.client.subscribe(
            topic=cfg['tdmq']['to_service_topic'],
            subscription_name=cfg['tdmq']['to_service_subscription_name'],
            message_listener=self.on_message,
            consumer_type=ConsumerType.Shared,
            initial_position=InitialPosition.Earliest,
        )
        print("waiting !!!!")
        
    def on_message(self, consumer, msg):
        
        """receive query data and run pipeline
        
        Args:
            msg: json.loads(msg.data()) = :
            {
                "service_name": self.service_name,
                "job_id": job_id,
                "parameter": 
                {
                    "in_obj_path": in_obj_path,
                    "in_condition_path": in_condition_path,
                    "in_data_type": in_data_type,
                    "out_dir": out_dir,
                    "vis_dir": vis_dir
                }
            }
        Returns:
            out_dict = {
                "service_name": self.service_name,
                "job_id": job_id,
                "success": False,
                "feedback": None,
                "raw_data": None,
                "job_id_dir": None,
                "out_dir": None
            }
        """
        job_id = ""
        msg_data = None
        try:
            msg_data = json.loads(msg.data())
            job_id = msg_data["job_id"]
            # assert len(job_id) > 0, "job id is empty"
            assert msg_data["service_name"] == self.service_name, f"unexpected service name {msg_data['service_name']}"

            self.run(msg_data["parameter"])
            success = True
            feedback = "ok"
            print(f"[ServiceConsumer] Processed job [{job_id}]")
        except Exception as e:
            feedback = "[server] " + str(e)
            success = False
            print(f"[ServiceConsumer] Error occurred when processing job [{job_id}]: {str(e)}\n" + traceback.format_exc())
        finally:
            # st()
            out_dict = {
                "service_name": self.service_name,
                "job_id": job_id,
                "success": success,
                "feedback": feedback,
                "raw_data": msg_data,
                "out_dir": msg_data["parameter"]["out_dir"],
                # "job_id_dir": msg_data["parameter"]["job_id_dir"]
            }
            out_data = json.dumps(out_dict)

            self.producer.send_async(
                out_data.encode("utf-8"),
                callback=self.send_callback,
            )
            consumer.acknowledge(msg)
        
    def send_callback(self, send_result, msg_id):
        print(f"[ServiceProducer] Sent msg_id [{msg_id}] with result [{send_result}]")
    
    def start(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.consumer.close()
            self.producer.close()
            self.client.close()
    
    def run(self, parameters):
        run_xyz2rgb(self.rmbg, self.controlnet, \
                    parameters["in_obj_path"], parameters["in_condition_path"], \
                    parameters["out_dir"], parameters["vis_dir"], \
                    parameters["in_data_type"])   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    
    service = Service(args.config)
    service.start()