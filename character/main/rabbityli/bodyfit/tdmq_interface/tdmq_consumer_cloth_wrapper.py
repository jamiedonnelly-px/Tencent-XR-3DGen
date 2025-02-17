import os, sys, json, uuid,time, pulsar, argparse, logging, threading
from _pulsar import ConsumerType, InitialPosition, LoggerLevel
from easydict import EasyDict as edict
from http.server import BaseHTTPRequestHandler, HTTPServer

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)

log_path = os.path.join(codedir, "logs", str(uuid.uuid4()))
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_path, "log.log"), level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.getLogger("pulsar").setLevel(logging.WARNING)

class MyRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"Hello, World!\n")

class LiveServer:
    def __init__(self, port=8986):
        self.server = HTTPServer(('0.0.0.0', port), MyRequestHandler)

    def run(self):
        logging.info(f'Starting live server on port {self.server.server_port}...\n')
        self.server.serve_forever()

class Consumer:
    def __init__(
            self,
            cfg_json='configs/tdmq.json',
            ip_port = 528
            ):
        """cloth_wrapper consumer(server) and backend producer.
        message producer(ClothWrapperProducer) -> pulsar queue -> ClothWrapperConsumer and send result to backend topic ->  BackendConsumer

        Args:
            cfg_json: relative name in codedir
        """
        ## 1. init cfg and grpc
        cfg_json = os.path.join(codedir, cfg_json)
        assert os.path.exists(cfg_json), f"can not find cfg_json {cfg_json}"
        with open(cfg_json, encoding='utf-8') as f:
            self.cfg_model = edict(json.load(f))
        self.log_dir = os.path.join(self.cfg_model.log_root_dir, time.strftime("%Y_%m_%d_%H_%M"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        ## 2. init cloth_wrapper consumer and backend producer
        tdmq_dict = self.cfg_model.tdmq
        self.client = pulsar.Client(
            authentication=pulsar.AuthenticationToken(tdmq_dict.token),# 已授权角色密钥
            service_url=tdmq_dict.service_url, # 服务接入地址
            logger=pulsar.ConsoleLogger(log_level=LoggerLevel.Warn)
            )
        
        self.run_py = os.path.join(codedir, "../webui/cloth_warpper.py")
        assert os.path.exists(self.run_py), f"can not find self.run_py = {self.run_py}"
        self.service_name = "cloth_wrapper"
        self.consumer = self.client.subscribe(
            # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
            topic=tdmq_dict.generation_topic,
            # 订阅名称
            subscription_name="cloth-wrapper-sub",
            # 设置监听
            message_listener=self.on_message,
            # 设置订阅模式为 Shared（共享）模式
            consumer_type=ConsumerType.Shared,
            properties={"cloth_wrapper": "TAGS"},
            # 配置从最早开始消费，否则可能会消费不到历史消息
            initial_position=InitialPosition.Earliest,
        )
        logging.info(f"init consumer done, topic={tdmq_dict.generation_topic}")
        # 客户端生产者
        self.producer_backend = self.client.create_producer(topic=tdmq_dict.backend_topic)
        logging.info(f"init producer_backend done, topic={tdmq_dict.backend_topic}")
        
        # 建一个存活端口
        self.live_server = LiveServer(ip_port)
        server_thread = threading.Thread(target=self.live_server.run)
        server_thread.start() 
        
        logging.info(f"begin listener")
                
        self.start()


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
                job_id = "ERROR_cloth_wrapper"
                logging.error(f"[ERROR] A serious interface error occurred. job_id could not be found.")
                print('[ERROR] msg', msg)
            
            job_id, success, result = job_id, False, f"error: {feedback}"
            print(f"Error occurred when query cloth_wrapper: {str(e)}")
        finally:
            logging.info(f"[receive] result of cloth_wrapper, job_id:{job_id}, success:{success}, result:{result}")
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
                properties={"cloth_wrapper_results": "TAGS"},
                # 业务key
                # partition_key='key1'
            )
            self.consumer.acknowledge(msg)

    def send_callback(self, send_result, msg_id):
        print("[send-backend-cloth-wrapper] send_result:{}  msg_id:{}".format(send_result, msg_id))
        
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
            "service_name": "cloth_wrapper",
            "parameter": {
                "job_id": job_id,
                "func_name": "cloth_wrapper",   # needed
                "in_obj_list": in_obj_list,   # needed
            }
                
        Returns:
            job_id
            suc_flag=T/F
            output_file= output_file(glb) or error feedback
        """
        st = time.time()
        # 1. parse data
        receive_data = json.loads(msg.data())
        service_name = receive_data.get("service_name", None)
        if service_name != self.service_name:
            print(f'error: receive_data service_name != {self.service_name}', receive_data)
            return False, f"invalid service_name: {service_name}"

        call_data = receive_data["parameter"]
        job_id = call_data["job_id"]
        logging.info(f"cloth_wrapper consumer receive job_id:{job_id}")
        print('call_data:', call_data)

        func_name = call_data.get("func_name", None)
        in_obj_list = call_data.get("in_obj_list", None)

        suc_flag = False
        output_path = ''
        if func_name == "cloth_wrapper":
            output_path = os.path.join(os.path.dirname(in_obj_list), "wrap_cloth.txt")
            if os.path.isfile(output_path):
                os.system(f"rm rf {output_path}")

            cmd_str = f"python {self.run_py} --lst_path {in_obj_list}"
            os.system(cmd_str)
            suc_flag = os.path.exists(output_path)
            logging.info(f"cloth_wrapper consumer run done job_id:{job_id}, suc_flag={suc_flag}, results:{output_path}")
            logging.info(f"use time:{time.time() - st}")
        else:
            logging.error(f"invalid func_name {func_name}")
            return job_id, False, f"invalid func_name {func_name}"
        
        return job_id, suc_flag, output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tqmd pulsar main consumer')
    parser.add_argument('--cfg_json', type=str, default='configs/tdmq.json')
    parser.add_argument('--ip_port', type=int, default=999)

    args = parser.parse_args()
    consumer = Consumer(args.cfg_json, args.ip_port)