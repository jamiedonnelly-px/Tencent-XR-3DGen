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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.getLogger("pulsar").setLevel(logging.WARNING)

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from dataset.utils_dataset import load_json
from pipe_safety_checker import SafetyCheckerPipe

class SafetyCheckerConsumer:

    def __init__(
            self,
            cfg_json='configs/safety_checker.json',
            device='cuda'):
        """consumer(server) and backend producer.
        message producer(TexGenProducer in main_call_texgen.py) -> pulsar queue -> TexGenConsumer and send result to backend topic ->  BackendConsumer

        Args:
            cfg_json: relative name in codedir/configs. Defaults to 'safety_checker.json'.
            
        """
        ## 1. init cfg and model
        self.load_cfg(cfg_json)
        self.device = device
        self.log_dir = os.path.join(self.cfg_model.log_root_dir, time.strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.pipe = SafetyCheckerPipe(self.cfg_model.safety_path, self.cfg_model.feature_extractor_path,
                                      device=device)

        ## 2. init tex_gen consumer and backend producer
        self.service_name = "safety_checker"
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
            subscription_name=f"{self.service_name}",
            # 设置监听
            message_listener=self.on_message,
            # 设置订阅模式为 Shared（共享）模式
            consumer_type=ConsumerType.Shared,
            properties={self.service_name: "TAGS"},
            # 配置从最早开始消费，否则可能会消费不到历史消息
            initial_position=InitialPosition.Earliest,
        )
        logging.info(f"init consumer done, topic={tdmq_dict.generation_topic}")

        self.producer_backend = self.client.create_producer(topic=tdmq_dict.backend_topic)
        logging.info(f"init producer_backend done, topic={tdmq_dict.backend_topic}")
        
        logging.info(f"begin listener")
                
        self.start()

    def load_cfg(self, cfg_json):
        """and self.cfg_model from json

        Args:
            cfg_json: abs path 
        """
        logging.info(f"input of load_run_cfg: cfg_json: {cfg_json}")
        assert os.path.exists(cfg_json), f"can not find valid cfg_json: {cfg_json}"
        cfg_dict = load_json(cfg_json)
    
        self.cfg_model = edict(cfg_dict)
        logging.info(f"load_run_cfg done ")
        return self.cfg_model

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
                job_id = f"ERROR_{self.service_name}"
                logging.error(f"[ERROR] A serious interface error occurred. job_id could not be found.")
                print('[ERROR] msg', msg)
            
            job_id, success, result = job_id, False, f"error: {feedback}"
            print(f"Error occurred when query {self.service_name}: {str(e)}")
        finally:
            ## return result to backend topic
            out_dict = {
                "service_name": self.service_name,
                "job_id": job_id,
                "flag": success,
                "result": result,
                "feedback": feedback
            }
            out_data = json.dumps(out_dict)

            self.producer_backend.send_async(
                # 消息内容
                out_data.encode("utf-8"),
                # 异步回调
                callback=self.send_callback,
                # 消息参数
                properties={f"{self.service_name}_results": "TAGS"},
                # 业务key
                # partition_key='key1'
            )
            self.consumer.acknowledge(msg)

    def send_callback(self, send_result, msg_id):
        print(f"[send-backend-{self.service_name}] send_result:{send_result}  msg_id:{msg_id}")
        
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


    def parse_and_run(self, msg):
        """receive query data and run pipeline

        Args:
            msg: json.loads(msg.data()) = :
            {
            "service_name": self.service_name,
            "parameter": {
                "job_id": job_id,
                "in_image_path": in_image_path,   # needed
            }
        Returns:
            job_id
            suc_flag=T/F
            results= safe or not
        """
        # 1. parse data
        receive_data = json.loads(msg.data())
        service_name = receive_data.get("service_name", None)
        if service_name != self.service_name:
            return False, f"invalid service_name: {service_name}"

        call_data = receive_data["parameter"]
        job_id = call_data["job_id"]
        logging.info(f"{self.service_name} consumer receive job_id:{job_id}")
        print('call_data:', call_data)

        in_image_path = call_data.get("in_image_path", None)

        results = self.pipe.check_image_is_safe(in_image_path)
        suc_flag = True
        logging.info(f"{self.service_name} consumer run done job_id:{job_id}, results:{results}")
        return job_id, suc_flag, results

def run_consumer(cfg_json):
    consumer = SafetyCheckerConsumer(cfg_json)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tqmd pulsar main consumer')
    parser.add_argument('--cfg_json', type=str, default='configs/safety_checker.json')
    args = parser.parse_args()

    cfg_json = args.cfg_json
    if not os.path.exists(cfg_json):
        cfg_json = os.path.join(codedir, cfg_json)
    assert os.path.exists(cfg_json)
    print('cfg_json', cfg_json)    
    
    run_consumer(cfg_json)

