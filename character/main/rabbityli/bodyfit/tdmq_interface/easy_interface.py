import os, sys, json, time, uuid, argparse, logging, threading
from concurrent.futures import ThreadPoolExecutor

codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)
from tdmq_client_cloth_wrapper import BackendConsumer, Producer

def init_job_id():
    return str(uuid.uuid4())

class EasyInterface:
    def __init__(
        self,
        client_cfg_json=os.path.join(codedir, 'configs/tdmq.json'),
    ):
        self.service_name = "cloth_wrapper"
        self.backend_consumer = BackendConsumer(client_cfg_json)
        consumer_thread = threading.Thread(target=self.backend_consumer.start_consumer)
        consumer_thread.start()
        time.sleep(0.1)
        logging.info("run BackendConsumer done")

        self.producer = Producer(client_cfg_json)
        logging.info(f"init producer done")

    def _blocking_call_query(self, job_id, query_func, timeout=300):
        """common blocking func

        Args:
            job_id: _description_
            query_func: _description_
            timeout: _description_. Defaults to 300.

        Returns:
            success_flag=T or F, 
            result_meshs list of obj path
        """
        query_job_ids = {job_id}
        query_func()
        time.sleep(0.1)

        start_time = time.time()
        result_dict = {}
        while query_job_ids:
            for job_id in list(query_job_ids):
                job_data = self.backend_consumer.redis_db.get(job_id)
                if job_data:
                    parse_data = json.loads(job_data)
                    result_dict[job_id] = parse_data
                    print(f"[[Received]] job_data for job_id {job_id}: {job_data.decode('utf-8')}")
                    query_job_ids.remove(job_id)
            time.sleep(0.1)
            if time.time() - start_time >= timeout:
                logging.error("Timeout: Exiting the loop.")
                break
        
        success_flag, result_meshs = self.parse_final_result_dict(job_id, result_dict)
        return success_flag, result_meshs
    
    def blocking_call(self, job_id, in_obj_list, timeout=300):
        """query text only mode

        Args:
            job_id(string) uuid
            in_obj_list(string): mesh绝对路径. raw obj/glb path with uv coord
            output_filename(string): 输出文件夹路径
            timeout(int): 单个任务的超时时间(s)
        Returns:
            success_flag=T or F, 
            result_meshs list of obj path
        """        
        logging.info(f"blocking_call_cloth_wrapper begin {job_id}")
        query_func = lambda: self.producer.interface_tdmq(
            job_id=job_id, in_obj_list=in_obj_list
        )
        logging.info(f"blocking_call_cloth_wrapper end {job_id}")
        return self._blocking_call_query(job_id, query_func, timeout)
    
    def parse_final_result_dict(self, job_id, final_result_dict):
        """Unify the final results into a consistent output as grpc: success_flag and result_meshs.

        Args:
            job_id: _description_
            final_result_dict: _description_

        Returns:
            success_flag=T or F, 
            result_meshs list of obj path
        """
        try:
            value_dict = final_result_dict[job_id]["cloth_wrapper"]
            assert self.service_name == value_dict["service_name"], f"invalid service_name != {self.service_name}"
            success_flag = value_dict["success"]
            result_meshs = value_dict["result"]
        
            return success_flag, result_meshs
        except Exception as e:
            print("parse_final_result_dict", final_result_dict)
            logging.error(
                f"[ERROR!!!!] parse_final_result_dict error, job_id: {job_id}, final_result_dict:{json.dumps(final_result_dict)}\n error:\n{e}"
            )
            return False, []
        
    def close(self):
        self.backend_consumer.close()
        self.producer.close()


def run_blocking_call(args):
    interface, job_id, in_obj_list, output_filename, timeout = args
    return interface.blocking_call(job_id,
                                    in_obj_list,
                                    output_filename=output_filename,
                                    timeout=timeout)


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='tqmd pulsar main consumer')
    parser.add_argument('--client_cfg_json',
                        type=str,
                        default=os.path.join(codedir, 'configs/tdmq.json'),
                        help='relative name in codedir/configs')
    args = parser.parse_args()
    ### prepare example data

    in_obj_list = "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/2e12b55d-f077-59b5-9ce3-9796db92a1bf/object_lst.txt"
    job_id = init_job_id()

    # 1. init
    interface = EasyInterface(args.client_cfg_json)

    # # 2. 堵塞式调用, 类似grpc
    success_flag, output_path = interface.blocking_call(job_id, in_obj_list)
    print("+++++++test++++: ", success_flag, output_path)
    assert success_flag
    interface.close()

    logging.info("test done")