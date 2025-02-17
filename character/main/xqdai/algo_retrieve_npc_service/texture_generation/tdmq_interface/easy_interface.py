from .main_call_texgen import TexGenProducer, BackendConsumer, init_job_id

import logging
import time
import json
import argparse
import os
import threading

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class TexgenInterface:

    def __init__(
        self,
        client_cfg_json="client_texgen.json",
        model_name="uv_mcwy",
        device="cuda",
    ):
        self.backend_consumer = BackendConsumer(client_cfg_json, model_name)
        consumer_thread = threading.Thread(target=self.backend_consumer.start_consumer)
        consumer_thread.start()
        time.sleep(0.1)
        logging.info(f"init backend_consumer done")

        self.producer = TexGenProducer(client_cfg_json, model_name, device=device)
        logging.info(f"init producer done")

    def blocking_call_query_text(
        self, in_mesh_path, in_prompts, in_mesh_key=None, out_objs_dir=None, timeout=300
    ):
        """query text only mode

        Args:
            in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
            in_prompts(string/ list of string): 文本提示,可以是字符串或字符串list
            in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
            out_objs_dir(string): 输出文件夹路径
            timeout(int): 单个任务的超时时间(s)
        Returns:
            result_dict: job_id:value
        """
        job_id = init_job_id()
        query_job_ids = set()
        query_job_ids.add(job_id)

        self.producer.interface_query_text(
            job_id,
            in_mesh_path,
            in_prompts,
            in_mesh_key=in_mesh_key,
            out_objs_dir=os.path.join(out_objs_dir, "query_key"),
        )
        time.sleep(0.1)

        start_time = time.time()
        result_dict = {}
        while query_job_ids:
            for job_id in list(query_job_ids):
                job_data = self.backend_consumer.redis_db.get(job_id)
                if job_data:
                    parse_data = json.loads(job_data)
                    print(parse_data)
                    result_dict[job_id] = parse_data
                    print(
                        f"[[Received]] job_data for job_id {job_id}: {job_data.decode('utf-8')}"
                    )
                    query_job_ids.remove(job_id)
            time.sleep(0.1)
            if time.time() - start_time >= timeout:
                logging.error("Timeout: Exiting the loop.")
                break
        return result_dict

    def close(self):
        self.backend_consumer.close()
        self.producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tqmd pulsar main consumer")
    parser.add_argument(
        "--client_cfg_json",
        type=str,
        default="client_texgen.json",
        help="relative name in codedir/configs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="uv_mcwy",
        help="select model. can be uv_mcwy, control_mcwy, imguv_mcwy, imguv_lowpoly, pipe_type_dataset",
    )
    args = parser.parse_args()
    ### prepare example data
    in_mesh_path = f"/aigc_cfs_gdp/Asset/designcenter/clothes/convert/mcwy2/remove_skin_mesh/meshes/Top/BR_TOP_1_F_T/BR_TOP_1_fbx2020.obj"
    in_prompts = "indian style"
    in_mesh_key = "BR_TOP_1_F_T"
    out_objs_dir = (
        f"/aigc_cfs_gdp/sz/server/tex_gen/pulsar_log/mcwy_debug_{args.model_name}"
    )

    # 1. init
    texgen_inferface = TexgenInterface(args.client_cfg_json, args.model_name)

    # 2. 堵塞式调用, 类似grpc
    result_dict = texgen_inferface.blocking_call_query_text(
        in_mesh_path, in_prompts, in_mesh_key=in_mesh_key, out_objs_dir=out_objs_dir
    )

    texgen_inferface.close()
    logging.info("test done")
