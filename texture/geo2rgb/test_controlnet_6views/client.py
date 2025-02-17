import pulsar
from _pulsar import ConsumerType, LoggerLevel
import uuid
import json
import logging
import time
import argparse
import os
import threading
import torch
from easydict import EasyDict as edict

import redis

codedir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def init_job_id():
    return str(uuid.uuid4())

class Zero123plusProducer:
    def __init__(
        self,
        cfg_json='config.json'
    ):
        self.load_cfg(cfg_json)
        self.device = torch.device("cuda:0")
        logging.info(f"load run cfg done from {cfg_json}")

        ## 2. init zero123plus consumer and backend producer
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(self.tdmq_config.token),# 已授权角色密钥
            service_url=self.tdmq_config.url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )

        self.producer = self.client.create_producer(topic=self.tdmq_config.topic)
        logging.info(f"init zero123plus producer done, topic={self.tdmq_config.topic}")

    ### external interface functions
    def query_zero123plus_with_image(
        self,
        job_id,
        in_image_path,
        out_save_dir=None,
        result_rmbg=True,
    ):
        """query image
        Args:
            job_id: job identity
            in_image_path: path of image for generate multiple view image
            out_save_dir: direction folder for saving generated multiview numpy
        Returns:
            send_flag: send succeed or failed
        """
        logging.info(f"begin interface_query_image of job_id:{job_id}")
        msg = self.make_message(job_id=job_id, in_image_path=in_image_path, out_save_dir=out_save_dir, result_rmbg=result_rmbg)
        try:
            self.send_messages(msg)
            return True
        except Exception as e:
            logging.warning(f"querry zero123plus server failed with: {e}")
            return False
    
    ###### Ignore the code that follows
    ### pulsar functions
    def send_messages(self, send_data: dict):
        self.producer.send_async(
            # 消息内容
            send_data.encode("utf-8"),
            # 异步回调
            callback=self.send_callback,
            # 消息参数
            properties={"zero123plus": "TAGS"},
            # 业务key
            # partition_key='key1'
        )

    def send_callback(self, send_result, msg_id):
        print("[send-zero123plus] send_result:{}  msg_id:{}".format(send_result, msg_id))

    def make_message(self, job_id, in_image_path, out_save_dir=None, result_rmbg=True):
        if out_save_dir is None:
            out_save_dir = self.log_dir
        assert job_id is not None and in_image_path is not None, "job_id or in_image_path must not be none!"

        send_dict = {
            "service_name": "zero123plus",
            "parameter": {
                "job_id": job_id,
                "in_image_path": in_image_path,
                "out_save_dir": out_save_dir,
                "wh_ratio": 0.8, # optional
                "cfg_scale": 3.0, # optional
                "step_num": 75, # optional
                "result_rmbg": result_rmbg, # optional
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

    ##### Internal functions
    def load_cfg(self, cfg_json):
        """
        Args:
            cfg_json: relative path in codedir/configs
        """
        logging.info(f"input of load_cfg:{cfg_json}")
        self.configs_dir = os.path.join(codedir, 'configs')
        cfg_json_ = os.path.join(self.configs_dir, cfg_json)
        if not os.path.exists(cfg_json_):
            logging.error(f"config path not exists: {cfg_json_}")
        assert os.path.exists(cfg_json_), f"can not find valid cfg_json: {cfg_json}"

        with open(cfg_json_, 'r') as fr:
            json_dict = edict(json.load(fr))
        
        self.model_config = json_dict.model_config
        self.tdmq_config = json_dict.tdmq_config

        self.model_name = self.model_config.model_name
        self.model_ckpt_dir = self.model_config.model_ckpt_dir
        self.log_dir = self.model_config.log_dir
        
        logging.info(f"load_cfg done with model_name:{self.model_name}")
        return


class Zero123plusBackendConsumer:
    def __init__(
        self,
        cfg_json="config.json",
    ):
        self.load_cfg(cfg_json)
        self.device = torch.device("cuda:0")
        logging.info(f"load run cfg done from {cfg_json}")

        ## 2. init tex_gen consumer and backend producer
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(self.tdmq_config.token),# 已授权角色密钥
            service_url=self.tdmq_config.url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )
        
        self.consumer = self.client.subscribe(
            # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
            topic=self.tdmq_config.backend_topic,
            # 订阅名称
            subscription_name='zero123plus_backend',
            # 设置监听
            message_listener=self.on_message,
            # 设置订阅模式为 Shared（共享）模式
            consumer_type=ConsumerType.Shared,
            consumer_name="结果消费者",
            properties={'zero123plus_backend': 'TAGS'},
        )

        self.redis_db = redis.StrictRedis(host=self.tdmq_config.redis_host,
                                          port=self.tdmq_config.redis_port,
                                          password=self.tdmq_config.redis_password)
        self.redis_db.set("debug", json.dumps({
            "service_name": "hello",
            "success": True,
        }))
        data = json.loads(self.redis_db.get("debug"))
        print('test db ', data)
        logging.info(f"init BackendConsumer done from {cfg_json}")
        self.stop_flag = False

    def parse_and_run(self, msg):
        """receive result of zero123plus
        Args:
            msg= result dict:
            {
                "job_id": job_id,
                "image_path": image_path,
                "wh_ratio":  0.8,
                "cfg_scale": 3.0,
                "step_num": 75,
                "image_npy_path": save_path,
                "normal_npy_path": None,
                "job_id_dir": job_id_dir ## for saving npy, obj, texture
            }
        Returns:
            job_id
            suc_flag=T/F
            parameter= result dict:
            {
                "job_id": job_id,
                "image_path": image_path,
                "wh_ratio":  0.8,
                "cfg_scale": 3.0,
                "step_num": 75,
                "image_npy_path": save_path,
                "normal_npy_path": None,
                "job_id_dir": job_id_dir ## for saving npy, obj, texture
            }
        """
        # 1. parse data
        receive_data = json.loads(msg.data())
        service_name = receive_data.get("service_name", None)
        if service_name != "zero123plus":
            return receive_data.get("job_id", "-1"), False, f"invalid service_name: {service_name}!=zero123plus"

        job_id = receive_data["job_id"]
        success = receive_data["flag"]
        result = receive_data["result"]
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
            print(f"Error occurred when receive result of zero123plus: {str(e)}")
        finally:
            logging.info(f"[receive] result of zero123plus, job_id:{job_id}, success:{success}, result:{result}")
            # 无论是哪个后端消费者消费, 都往redis里写数据, 保证数据不丢；然后后端生产者可以根据job id去查询结果

            job_data = self.redis_db.get(job_id)
            if job_data is not None:
                job_data_dict = json.loads(job_data)
                job_data_dict["z123"] = {
                                        "service_name": "z123",
                                        "success": success,
                                        "parameter": result
                                    }
            else:
                job_data_dict = {
                            "z123":{
                                    "service_name": "z123",
                                    "success": success,
                                    "parameter": result
                            }
                        }
            self.redis_db.set(job_id, json.dumps(job_data_dict))

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


    ##### Internal functions
    def load_cfg(self, cfg_json):
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
        self.tdmq_config = json_dict.tdmq_config

        self.model_name = self.model_config.model_name
        self.model_ckpt_dir = self.model_config.model_ckpt_dir
        self.log_dir = self.model_config.log_dir
        
        logging.info(f"load_cfg done with self.model_name:{self.model_name}")
        return


    def start_consumer(self):
        self.start()

    def close(self):
        self.stop_flag = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tqmd pulsar client consumer')
    parser.add_argument('--cfg_json',
                        type=str,
                        default='zero123plus_stable.json',
                        help='relative name in codedir/configs')
    args = parser.parse_args()

    ### run backend consumer, receive result of zero123plus
    backend_consumer = Zero123plusBackendConsumer(args.cfg_json)
    consumer_thread = threading.Thread(target=backend_consumer.start_consumer)
    consumer_thread.start()
    time.sleep(1)
    logging.info("run backend consumer done")

    ### prepare example data
    # image_path = "/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation/mario.png"
    image_path = "/aigc_cfs_gdp/neoshang/data/validation/mario.png"

    # 1. init producer
    producer = Zero123plusProducer(args.cfg_json)
    query_job_ids = set()

    ### query_zero123plus_with_image
    for i in range(3):
        job_id = init_job_id()
        query_job_ids.add(job_id)
        producer.query_zero123plus_with_image(
                job_id,
                image_path,
                out_save_dir=None,
            )
        time.sleep(0.2)


    # 3. query results. backend同步用法: 从数据库里轮循结果, 后端形成一个闭环.
    print('query_job_ids ', query_job_ids)
    start_time = time.time()
    timeout = 50 * len(query_job_ids)
    while query_job_ids:
        for job_id in list(query_job_ids):
            job_data = backend_consumer.redis_db.get(job_id)
            if job_data:
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