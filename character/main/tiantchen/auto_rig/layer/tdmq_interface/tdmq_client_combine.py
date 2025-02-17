import os, json, time, redis, pulsar, logging
from _pulsar import ConsumerType, LoggerLevel
from easydict import EasyDict as edict

codedir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

### utils func
def load_json(in_file):
    with open(in_file, encoding='utf-8') as f:
        data = json.load(f)
    return data

class CombineProducer:
    def __init__(
        self,
        client_cfg_json="configs/tdmq_combine.json",
    ):
        client_cfg_json = os.path.join(codedir, client_cfg_json)
        assert os.path.exists(client_cfg_json), f"can not find client_cfg_json {client_cfg_json}"
        with open(client_cfg_json, encoding='utf-8') as f:
            self.cfg_model = edict(json.load(f))
        tdmq_dict = edict(self.cfg_model).tdmq
        logging.info(f"load run cfg done from {client_cfg_json}")

        self.service_name = "combine"
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),# 已授权角色密钥
            service_url=tdmq_dict.service_url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )
        self.producer = self.client.create_producer(topic=tdmq_dict.generation_topic)
        logging.info(f"init CombineProducer  done, topic={tdmq_dict.generation_topic}")

    ### external interface functions
    def interface_tdmq_combine(
        self,
        job_id,
        in_mesh_path,
        output_mesh_filename,
    ):
        """combine

        Args:
            job_id(string): uuid
            in_mesh_path(string): raw obj
            output_mesh_filename(string):  out prefix name: fbx
        Returns:
            send_flag: send succeed or failed
        """
        logging.info(f"begin interface_tdmq_combine of job_id:{job_id}")
        send_flag = self.call_combine(
            job_id=job_id,
            func_name="combine",
            in_mesh_path=in_mesh_path,
            output_mesh_filename=output_mesh_filename,
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
            properties={"combine": "TAGS"},
            # 业务key
            # partition_key='key1'
        )

    def send_callback(self, send_result, msg_id):
        print("[send-combine] send_result:{}  msg_id:{}".format(send_result, msg_id))

    def close(self):
        logging.info("closing....")
        time.sleep(3)
        self.producer.close()
        self.client.close()
        logging.info("closed")

    ### core function
    def call_combine(
        self,
        job_id,
        func_name="combine",
        in_mesh_path=None,
        output_mesh_filename=None,
    ):
        """send message, call combine

        Args:
            job_id(string): uuid
            func_name(string): combine
            in_mesh_path(string): 
            output_mesh_filename(string): 
        Returns:
            send flag=T/F
        """
        try:
            logging.info(
                f"begin call_combine, job_id {job_id} with func_name {func_name} and input_path {in_mesh_path}, output_mesh_filename={output_mesh_filename}"
            )

            ### begin
            send_dict = {
                "service_name": self.service_name,
                "parameter": {
                    "job_id": job_id,
                    "func_name": func_name,   # needed
                    "in_mesh_path": in_mesh_path,   # needed
                    "output_mesh_filename": output_mesh_filename, # needed
                },
            }
            send_data = json.dumps(send_dict)

            self.send_messages(send_data)

            logging.debug(str(send_data))
        except Exception as e:
            print(f"run combine failed! {e}")
            return False
        return True


class CombineBackendConsumer:
    def __init__(
        self,
        client_cfg_json='configs/tdmq_combine.json',
    ):
        client_cfg_json = os.path.join(codedir, client_cfg_json)
        print(client_cfg_json)
        assert os.path.exists(client_cfg_json), f"can not find client_cfg_json {client_cfg_json}"
        with open(client_cfg_json, encoding='utf-8') as f:
            self.cfg_model = edict(json.load(f))
        tdmq_dict = edict(self.cfg_model).tdmq

        self.service_name = "combine"
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),  # 已授权角色密钥
            service_url=tdmq_dict.service_url,  # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn))
        self.consumer = self.client.subscribe(
            # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
            topic=tdmq_dict.backend_topic,
            # 订阅名称
            subscription_name='combine-backend-sub',
            # 设置监听
            message_listener=self.on_message,
            # 设置订阅模式为 Shared（共享）模式
            consumer_type=ConsumerType.Shared,
            consumer_name="结果消费者",
            properties={'combine_backend-': 'TAGS'},
        )

        self.redis_db = redis.StrictRedis(host=tdmq_dict.redis_host,
                                          port=tdmq_dict.redis_port,
                                          password=tdmq_dict.redis_password)
        # self.redis_db.set("debug", json.dumps({
        #     "service_name": "hello",
        #     "success": True,
        # }))
        # data = json.loads(self.redis_db.get("debug"))
        # print('test db ', data)
        self.stop_flag = False
        logging.info(f"init CombineBackendConsumer done from {client_cfg_json} and topic {tdmq_dict.backend_topic}")


    def parse_and_run(self, msg):
        """receive result of combine

        Args:
            msg: json.loads(msg.data()) = :
            out_dict = {
                "service_name": "combine",
                "job_id": job_id,   # str uuid
                "flag": success,    # T/F
                "result": result,   # 输出的结果路径或报错信息
                "feedback": feedback # 报错信息(如果有)
            }
        Returns:
            job_id
            suc_flag=T/F
            results= 输出的结果路径或报错信息
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
            print(f"Error occurred when receive result of combine: {str(e)}")
        finally:
            logging.info(f"[receive] result of combine, job_id:{job_id}, success:{success}, result:{result}")
            # 无论是哪个后端消费者消费, 都往redis里对应job_id写数据, 保证数据不丢；然后后端生产者可以根据job id去查询结果
            value_dict = {"service_name": self.service_name, "success": success, "result": result}

            job_data = self.redis_db.get(job_id)
            if job_data is not None:
                job_data_dict = json.loads(job_data)
                job_data_dict[self.service_name] = value_dict
            else:
                job_data_dict = {self.service_name: value_dict}
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