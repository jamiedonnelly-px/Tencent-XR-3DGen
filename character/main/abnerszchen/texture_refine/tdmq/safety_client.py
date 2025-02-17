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

class SafetyCheckerProducer:

    def __init__(
        self,
        client_cfg_json="configs/client_safety_checker.json",
        device='cuda',
    ):
        self.load_run_cfg(client_cfg_json)
        self.device = device
        self.service_name = "safety_checker"
        logging.info(f"load run cfg done from {client_cfg_json}")

        tdmq_dict = self.run_cfg.tdmq
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),# 已授权角色密钥
            service_url=tdmq_dict.service_url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )
        self.producer = self.client.create_producer(topic=tdmq_dict.generation_topic)
        logging.info(f"init safety producer done, topic={tdmq_dict.generation_topic}")

    ### external interface functions
    def interface_check_image_safety(
        self,
        job_id,
        in_image_path,
    ):
        """query text only mode

        Args:
            job_id(string): uuid
            in_image_path(string): image绝对路径. cfs gdp or cos4
        Returns:
            send_flag: send succeed or failed
        """
        try:
            logging.info(
                f"begin interface_check_image_safety, job_id {job_id} with in_image_path {in_image_path}"
            )
            ### begin
            send_data = self.make_message(
                job_id=job_id,
                in_image_path=in_image_path
            )
            self.send_messages(send_data)

            logging.debug(str(send_data))
        except Exception as e:
            print(f"run interface_check_image_safety failed! {e}")
            return False
        return True

  
    ###### Ignore the code that follows
    ### pulsar functions
    def send_messages(self, send_data: dict):
        self.producer.send_async(
            # 消息内容
            send_data.encode("utf-8"),
            # 异步回调
            callback=self.send_callback,
            # 消息参数
            properties={self.service_name: "TAGS"},
            # 业务key
            # partition_key='key1'
        )

    def send_callback(self, send_result, msg_id):
        print("[send-tex] send_result:{}  msg_id:{}".format(send_result, msg_id))

    def make_message(
            self,
            job_id="",
            in_image_path=None,
    ):
        send_dict = {
            "service_name": self.service_name,
            "parameter": {
                "job_id": job_id,
                "in_image_path": in_image_path,
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
    def load_run_cfg(self, cfg_json):
        """and self.run_cfg from json

        Args:
            cfg_json: abs path or relative name in codedir/configs
            model_name: model key. Defaults to None.
        """
        logging.info(f"input of load_run_cfg: cfg_json: {cfg_json}")
        assert os.path.exists(cfg_json), f"can not find valid cfg_json: {cfg_json}"
        cfg_dict = load_json(cfg_json)
    
        self.run_cfg = edict(cfg_dict)
        logging.info(f"load_run_cfg done ")
        return self.run_cfg


class SafetyCheckerConsumer:

    def __init__(
        self,
        client_cfg_json="configs/client_safety_checker.json",
    ):
        assert os.path.exists(client_cfg_json), f"can not find {client_cfg_json}"
        tdmq_dict = edict(load_json(client_cfg_json)).tdmq

        self.service_name = "safety_checker"
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),  # 已授权角色密钥
            service_url=tdmq_dict.service_url,  # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn))
        self.consumer = self.client.subscribe(
            # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
            topic=tdmq_dict.backend_topic,
            # 订阅名称
            subscription_name=f'backend-{ self.service_name}-sub',
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
        logging.info(f"init BackendConsumer done from {client_cfg_json}")


    def parse_and_run(self, msg):
        """receive result of pipe consumer

        Args:
            msg: json.loads(msg.data()) = :
            out_dict = {
                "service_name": "safety_checker",
                "job_id": job_id,   # str uuid
                "flag": success,    # T/F
                "result": result,   # T/F: safe or not
                "feedback": feedback # 报错信息(如果有)
            }
        Returns:
            job_id
            suc_flag=T/F
            results= T/F: safe or not
        """
        # 1. parse data
        receive_data = json.loads(msg.data())
        service_name = receive_data.get("service_name", None)
        if service_name != self.service_name:
            return receive_data.get("job_id", "-1"), False, f"invalid service_name: {service_name}!={self.service_name}"

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
            # 代码应该避免进到这里， 无论任务是否成功都应该要跑完parse_and_run
            # 跑到这里一般是因为传的字典里少了某些parse_and_run需要的key， 在写代码的时候就要避免
            # 无论如何都要拿到job_id，否则不能往redis里写， 导致任务丢失
            job_id = json.loads(msg.data()).get("job_id", "-1")            
            job_id, success, result = job_id, False, ""
            print(f"Error occurred when receive result of {self.service_name}: {str(e)}")
        finally:
            logging.info(f"[receive] result of {self.service_name}, job_id:{job_id}, success:{success}, result:{result}")
            # 无论是哪个后端消费者消费, 都往redis里对应job_id写数据, 保证数据不丢；然后后端生产者可以根据job id去查询结果
            value_dict = {"service_name": f"{self.service_name}", "success": success, "result": result}

            job_data = self.redis_db.get(job_id)
            if job_data is not None:
                job_data_dict = json.loads(job_data)
                job_data_dict[f"{self.service_name}"] = value_dict
            else:
                job_data_dict = {f"{self.service_name}": value_dict}
            print('job_data_dict ', job_data_dict)
            
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

    def start_consumer(self):
        self.start()
    
    def close(self):
        self.stop_flag = True

def test_safety_client(client_cfg_json):

    ### run backend consumer, receive result of pipe
    backend_consumer = SafetyCheckerConsumer(client_cfg_json)
    consumer_thread = threading.Thread(target=backend_consumer.start_consumer)
    consumer_thread.start()
    time.sleep(1)
    logging.info("run backend consumer done")

    ### prepare example data
    in_image_path = f"/aigc_cfs_gdp/sz/data/safety/sexy1.jpeg"

    # 1. init producer
    producer = SafetyCheckerProducer(client_cfg_json)
    query_job_ids = set()

    # 2. run producer, call query
    try_count = 10
    for i in range(try_count):
        job_id = init_job_id()
        query_job_ids.add(job_id)
        producer.interface_check_image_safety(
            job_id,
            in_image_path
        )
        time.sleep(0.2)

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
    backend_consumer.close()

    logging.info("test done, you can shut done with ctrl+c")

    # consumer_thread.join()
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tqmd pulsar main consumer')
    parser.add_argument('--client_cfg_json',
                        type=str,
                        default='configs/client_safety_checker.json',
                        help='relative name in codedir')
    args = parser.parse_args()
    
    client_cfg_json = args.client_cfg_json
    if not os.path.exists(client_cfg_json):
        client_cfg_json = os.path.join(codedir, client_cfg_json)
    assert os.path.exists(client_cfg_json)
    print('client_cfg_json', client_cfg_json)

    test_safety_client(client_cfg_json)