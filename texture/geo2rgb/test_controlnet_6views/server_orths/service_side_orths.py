import sys
import os
import redis
# sys.path.append(os.path.abspath("/aigc_cfs_2/zacheng/MMD_NPU_dmmd/"))
# sys.path.append(os.path.abspath("/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus"))
sys.path.append(os.path.abspath("/aigc_cfs/xibinsong/code/zero123plus_control/zero123plus_gray"))
# from src.diffusers import MMDiffusionDepthToRgbImagePipeline as MMDiffusionDepthToRgbImagePipeline
# from src.diffusers import MMDiffusionNormalToRgbImagePipeline_dino_vit_4views_cuda as MMDiffusionDepthToRgbImagePipeline
from server_orths.inference_orths import run_xyz2rgb
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

class Service:
    
    def __init__(self, cfg_json):
        
        with open(cfg_json) as json_file:
            cfg = json.load(json_file)
            
        
        self.service_name = cfg['service_name']
        # self.in_pose_npy = cfg['data']['pose_npy']

        # print("loading pretrained models !!!")
        
        # self.num_tasks_new = cfg['data']['num_tasks_new']
        # self.pipeline_new = MMDiffusionDepthToRgbImagePipeline.from_pretrained(cfg['data']['pipeline_new'], num_views=4, num_tasks=self.num_tasks_new, 
            # torch_dtype={"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg['data']['precision']])
        # self.pipeline_new.to("cuda")
        # self.pipeline_new.enable_xformers_memory_efficient_attention()

        # print("loading dino features !!!")

        # self.dino_image_processor = ViTImageProcessor.from_pretrained('/aigc_cfs/model/dino-vitb16')
        # self.dino_model = ViTModel.from_pretrained('/aigc_cfs/model/dino-vitb16')

        # print("loading vit features !!!")
        # self.vit_model_name = "/aigc_cfs/model/vit-base-patch16-224"
        # self.vit_model = ViTModel.from_pretrained(self.vit_model_name)
        # self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained(self.vit_model_name)

        # print("finish loading models !!!")

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
                    "in_pose_npy": None,
                    "in_rgb_npy": in_rgb_npy,
                    "out_dir": out_dir,
                    "vis_dir": vis_dir,
                    "job_id_dir": job_id_dir
                    "seed": 0,
                    "cfg": 1.0
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
            # if msg_data["parameter"]["in_pose_npy"] is None:
            #     msg_data["parameter"]["in_pose_npy"] = self.in_pose_npy
            #if msg_data["parameter"]["condition_img"] is None:

            redis_db = redis.StrictRedis(host="21.29.224.143", port="6379", password="Guokong615")
            # job_id = "20fe9a8a-d0d5-42fc-a31c-f816a7945bcd"
            data = json.loads(redis_db.get(job_id))
            # print('data', data)
            print(data.keys())
            if 'z123' in data.keys():
                print("cond image path: ", data['z123']['parameter']['image_path_list'][0])
                # print("cond image path: ", data['z123']['parameter']['image_npy_path_rescale'])
                # breakpoint()
                condition_img_path = data['z123']['parameter']['image_path_list'][0]
                in_data_type = 'z123'
            elif ('mesh2image' in data.keys()) and ('mv2mesh' in data.keys()):
                print("input type is mv-2-8-view images !")
                condition_img_path = data['mesh2image']['result']
                in_data_type = 'mesh2image'
            elif ('mv2mesh' in data.keys()) and ('mesh2image' not in data.keys()):
                print("input type is 3 views !")
                condition_img_path = data['mv2mesh']['result']['parameter']['job_id_dir']
                condition_img_path = os.path.join(condition_img_path, "direct_stack.npy")
                in_data_type = 'mv2mesh'
            else:
                print("invalid condition image path !!!")
                print("can not find valid keys in redis, one of z123 and mv2mesh must exist !")
                raise ValueError("invalid condition image path, can not find valid keys in redis, one of z123 and mv2mesh must exist !")

            # condition_img = data['z123']['parameter']['image_path_list'][0]
            condition_img = condition_img_path
            # condition_img = data['z123']['parameter']['image_npy_path_rescale']
            msg_data["parameter"].update({"condition_img": condition_img})
            msg_data["parameter"].update({"in_data_type": in_data_type})

            # if "mv2mesh" in data:
            #     obj_gen_type = "mv2mesh"
            # elif "lrm" in data.keys():
            #     obj_gen_type = "lrm"
            # else:
            #     obj_gen_type = "lrm"
            # print(obj_gen_type)
            # # breakpoint()
            # msg_data["parameter"].update({"obj_gen_type": obj_gen_type})

            # print(msg_data["parameter"])
            # breakpoint()

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
                "job_id_dir": msg_data["parameter"]["job_id_dir"]
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
        print(f'running pipeline: {parameters.get("pipeline", "none")}')
        if parameters.get("pipeline", "new") == "new":
            # run_d2rgb(self.pipeline_new, parameters["in_obj_path"], parameters["in_pose_npy"], parameters["in_rgb_npy"], parameters["out_dir"], parameters["vis_dir"], parameters["seed"], parameters["cfg"])   
            # run_xyz2rgb(self.pipeline_new, self.dino_image_processor, self.dino_model, \
            #             self.vit_feature_extractor, self.vit_model,  \
            #             parameters["in_obj_path"], parameters["in_rgb_npy"], \
            #             parameters["out_dir"], parameters["vis_dir"], \
            #             parameters["seed"], parameters["cfg"])
            print("condition_img: ", parameters["condition_img"])
            print("out_dir: ", parameters["out_dir"])
            print("vis_dir: ", parameters["vis_dir"])
            print("in_data_type: ", parameters["in_data_type"]) 
            # run_xyz2rgb(self.pipeline_new, self.dino_image_processor, self.dino_model, \
            #             self.vit_feature_extractor, self.vit_model,  \
            #             parameters["in_obj_path"], parameters["condition_img"], \
            #             parameters["out_dir"], parameters["vis_dir"], \
            #             parameters["obj_gen_type"], \
            #             parameters["seed"], parameters["cfg"])   

            # run_xyz2rgb(self.pipeline_new, self.dino_image_processor, self.dino_model, \
            #             self.vit_feature_extractor, self.vit_model,  \
            #             parameters["in_obj_path"], parameters["condition_img"], \
            #             parameters["out_dir"], parameters["vis_dir"], \
            #             parameters["seed"], parameters["cfg"])   

            run_xyz2rgb(parameters["in_obj_path"], parameters["condition_img"], \
                        parameters["out_dir"], parameters["vis_dir"], \
                        parameters["in_data_type"], \
                        parameters["seed"], parameters["cfg"])   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    
    service = Service(args.config)
    service.start()