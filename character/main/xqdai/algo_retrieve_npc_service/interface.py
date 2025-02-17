from gradio_client import Client
import sys

sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/CLIP_img_gen")
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/server_bakend/apps")
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/motion")
sys.path.append("/aigc_cfs/tinatchen/auto_rig/layer/change_hair")
from cos import CosClient
from single_retrieve import retrive_single_txt
import uuid
import time
import json
import requests
from retrieval import keys_dict
import os

sys.path.append("/aigc_cfs_2/xiaqiangdai/project/cloth_wrap/webui")
from shape_retrival_rerank import shape_text_retrieve
from base_body_map import base_body_map


import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer, ThreadPoolServer
from gradio_client import Client

rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
rpyc_config["sync_request_timeout"] = None
rpyc_config["allow_public_attrs"] = True
import argparse
import threading
import zlib
import pickle
from extract_entity import extract_entity_all
import ujson
import logging
from datetime import datetime
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/texture_generation/grpc_interface")
from client_texgen import TexGenClient, init_job_id
from animation_interface import animation, animation_text, animation_gif
from distribute_logging import *
from shader.Run_Render import *
from is_change_ok import is_change_ok

parser = argparse.ArgumentParser(description="启动所有RPC")
parser.add_argument("-H1", "--hostname1", default="0.0.0.0", help="外部设置ip")

InMyweb = False

json_path = "/aigc_cfs_2/xiaqiangdai/project/objaverse_retrieve/20240711_ruku_ok.json"
gdp_json_path="/aigc_cfs_gdp/list/active_list2/layered_data/20240711/20240711_ruku_ok_gdp.json"
model_save_folder = "/mnt/aigc_bucket_4/pandorax/retrieveNPC_save/"

global g_json_data
g_json_data = keys_dict(json_path)
from gradio_client import Client


if not os.path.exists("/root/envs/auto_rig"):
    print("cp -r /mnt/aigc_cfs_cq/xiaqiangdai/envs/auto_rig.tar.gz /root/envs/")
    print("cd /root/envs/ && tar -zxvf auto_rig.tar.gz")
    os.system(
        "mkdir /root/envs/ && cp -r /mnt/aigc_cfs_cq/xiaqiangdai/envs/auto_rig.tar.gz /root/envs/"
    )
    os.system("cd /root/envs/ && tar -zxvf auto_rig.tar.gz")


def get_size(obj):
    size = sys.getsizeof(obj)

    if isinstance(obj, list):
        size += sum(get_size(item) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_size(k) + get_size(v) for k, v in obj.items())
    elif isinstance(obj, (tuple, set)):
        size += sum(get_size(item) for item in obj)

    return size


def is_english_or_chinese(s):
    for ch in s:
        name = unicodedata.name(ch)
        if "CJK UNIFIED" in name or "CJK COMPATIBILITY" in name:
            return "Chinese"
    return "English"


def remove_files_older_than_n_days(directory, days):
    current_time = datetime.datetime.now()
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_creation_time = datetime.datetime.fromtimestamp(
                os.path.getctime(file_path)
            )
            file_age = (current_time - file_creation_time).days
            if file_age > days:
                os.remove(file_path)
                print(f"Removed file: {file_path}")

def prompt_preprocess(strs):
    
    for i,s in enumerate(strs):
        if s=='---':
            strs[i]=''

    replace_strs = {'short skirt':'skirt'}
    for key in replace_strs.keys():
        strs[2] = strs[2].replace(key,replace_strs[key])
    
    
    replace_strs = {'short':'short hair','long':'long hair'}
    for key in replace_strs.keys():
        if replace_strs[key] not in strs[0]:  
            strs[0] = strs[0].replace(key,replace_strs[key])

    return strs

genders = ["male", "female"]
part_keys = ["hair", "top", "trousers", "shoe", "outfit", "others"]


class RPCService(Service):

    def __init__(self):
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day

        self.log_file_path = (
            f"/aigc_cfs_2/xiaqiangdai/data/rpc_logs_normal/{year}_{month}_{day}.log"
        )
        self.logger = DistributedFileLogger("RPCService_logger", self.log_file_path)
        self.sleepTime = 10
        self.processNum = 0
        self.uuid = None

    def on_connect(self, conn):
        # 连接建立时的回调函数
        print("rpyc 开始连接...")
        # conn._config['uuid'] = uuid.uuid4()
        self.processNum += 1
        if self.processNum > 100:
            remove_files_older_than_n_days(
                "/aigc_cfs_2/xiaqiangdai/data/rpc_logs_normal", 10
            )
            now = datetime.now()
            year = now.year
            month = now.month
            day = now.day
            self.log_file_path = (
                f"/aigc_cfs_2/xiaqiangdai/data/rpc_logs_normal/{year}_{month}_{day}.log"
            )
            self.logger = DistributedFileLogger("RPCService_logger", self.log_file_path)
            self.processNum = 0

    def on_disconnect(self, conn):
        # 连接断开时的回调函数
        print("rpyc 断开连接...")

    def exposed_set_uuid(self, uuid):
        self.uuid = uuid

    def wrap_cloth(self, mesh_output_path, paths_temp):

        # paths_temp_in = {"path": paths_temp, "body_attr": [gender, shape],"hair_color":entity[0]}

        gender = paths_temp["body_attr"][0]
        shape_promt = paths_temp["body_attr"][1]
        try:
            body_attr = shape_text_retrieve(gender, shape_promt)
        except:
            self.logger.error(f"{self.uuid} shape_text_retrieve exception")
        paths_temp["body_attr"] = body_attr
        paths_temp["shape_promt"] = shape_promt

        obj_lst = os.path.join(mesh_output_path, "object_lst.txt")

        json_object = json.dumps(paths_temp, indent=4)

        self.logger.info(f"{self.uuid} json_object:{json_object}")
        with open(obj_lst, "w") as f:
            f.write(json_object)

        cmd = [
            "/root/envs/auto_rig/bin/python",
            "/aigc_cfs_2/xiaqiangdai/project/cloth_wrap/webui/cloth_warpper.py",
            "--lst_path",
            obj_lst,
        ]

        cmd = " ".join(cmd)
        os.system(cmd)

        return mesh_output_path

    def exposed_cos_upload(self, glb_local_path):
        guid = uuid.uuid4().hex
        new_glb_name = f"{guid}/{os.path.basename(glb_local_path)}"
        cos_client = CosClient()
        remote_file_path = os.path.join(
            cos_client.retrieve_NPC_dir_remote, new_glb_name
        )
        glb_url = cos_client.upload(glb_local_path, remote_file_path)
        self.logger.info(f"{self.uuid} glb_url:{glb_url}")
        json_str = ujson.dumps(glb_url)
        return json_str
    
    def exposed_cos_upload_remote(self,glb_local_path,remote_file_path):
        try:
            cos_client = CosClient()
            glb_url = cos_client.upload(glb_local_path, remote_file_path)
            self.logger.info(f"{self.uuid} glb_local_path:{glb_local_path} remote_file_path:{remote_file_path} glb_url:{glb_url}")
            json_str = ujson.dumps(glb_url)
            return json_str
        except Exception as e:
            print(e)
            return None

    def exposed_texReplace(self, prompt, path, key):
        texclient = TexGenClient(
            client_cfg_json="/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/texture_generation/configs/client_texgen.json"
        )
        job_id = init_job_id()
        out_mesh_paths_query_key = texclient.webui_query_text(
            job_id,
            path,
            prompt,
            key,
            out_objs_dir=model_save_folder+"texture_save",
        )
        return ujson.dumps(out_mesh_paths_query_key[0])

    def exposed_txt_retrive_npc(self, entity, description):
        """
        npc text retrieve
        input:txt list([hair,top,trousers,shoe,outfit,others,gender])
        output info:[hair image list,top image list,trousers image list,shoe image list,outfit image list,others image list,
                    hair glb list,top glb list,trousers glb list,shoe glb list,outfit glb list,others glb list,
                    hair key list,top key list,trousers key list,shoe key list,outfit key list,others key list]
        """

        start_time = time.time()
        strs = entity[:6]

        self.logger.info(f"{self.uuid} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]

        strs = prompt_preprocess(strs)

        self.logger.info(f"{self.uuid} {description} {entity}")
        self.logger.info(f"{self.uuid} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4] != "":
            suit_enale = True

        gender = entity[-1]
        if gender not in genders:
            self.logger.info(f"{self.uuid} txt_retrive_npc gender error")
            return ujson.dumps([None])

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender + "_" + part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{self.uuid} key:{s} None_num:{None_num}")

            if None_num == 3 and key_id!=0 and key_id!=3:
                keys = retrive_single_txt(
                    description[:70], gender + "_" + part_keys[key_id]
                )

            self.logger.info(f"{self.uuid} keys:{keys}")
            for i in range(len(keys)):
                if keys[i] != None:
                    # print(keys)
                    img_paths_out.append(g_json_data[keys[i]]["Preview"])
                    glb_paths_out.append(g_json_data[keys[i]]["GLB_Mesh"])
                else:
                    img_paths_out.append(None)
                    glb_paths_out.append(None)
                keys_out.append(keys[i])

        final_out.append([glb_paths_out[0], glb_paths_out[1], glb_paths_out[2]])
        final_out.append([glb_paths_out[3], glb_paths_out[4], glb_paths_out[5]])
        final_out.append([glb_paths_out[6], glb_paths_out[7], glb_paths_out[8]])
        final_out.append([glb_paths_out[9], glb_paths_out[10], glb_paths_out[11]])
        final_out.append([glb_paths_out[12], glb_paths_out[13], glb_paths_out[14]])
        final_out.append([glb_paths_out[15], glb_paths_out[16], glb_paths_out[17]])

        final_out.append([keys_out[0], keys_out[1], keys_out[2]])
        final_out.append([keys_out[3], keys_out[4], keys_out[5]])
        final_out.append([keys_out[6], keys_out[7], keys_out[8]])
        final_out.append([keys_out[9], keys_out[10], keys_out[11]])
        final_out.append([keys_out[12], keys_out[13], keys_out[14]])
        final_out.append([keys_out[15], keys_out[16], keys_out[17]])

        final_out.append([img_paths_out[0], img_paths_out[1], img_paths_out[2]])
        final_out.append([img_paths_out[3], img_paths_out[4], img_paths_out[5]])
        final_out.append([img_paths_out[6], img_paths_out[7], img_paths_out[8]])
        final_out.append([img_paths_out[9], img_paths_out[10], img_paths_out[11]])
        final_out.append([img_paths_out[12], img_paths_out[13], img_paths_out[14]])
        final_out.append([img_paths_out[15], img_paths_out[16], img_paths_out[17]])
        final_out.append(entity[-1])  # gender
        final_out.append(suit_enale)
        if keys_out[0]!=None:
            final_out.append(entity[-2]) #hair color 20
        else:
            final_out.append(None)

        # list_bytes = pickle.dumps(final_out)
        # compressed_data = zlib.compress(list_bytes)

        # print(f"compressed_data:{get_size(compressed_data)}")
        # print(f"ori_data:{get_size(final_out)}")
        json_str = ujson.dumps(final_out)

        end_time = time.time()
        self.logger.info(
            f"{self.uuid} txt_retrive cost time: {end_time - start_time} s"
        )

        return json_str

    def exposed_long_txt_retrive_npc(self, long_prompt):
        """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output: similar like upper case
        output info:[hair image list,top image list,trousers image list,shoe image list,outfit image list,others image list,
                    hair glb list,top glb list,trousers glb list,shoe glb list,outfit glb list,others glb list,
                    hair key list,top key list,trousers key list,shoe key list,outfit key list,others key list]
        """
        start_time = time.time()
        # strs = extract_entity(long_prompt).split('/')
        entity, description = extract_entity_all(long_prompt)

        end_time = time.time()
        self.logger.info(
            f"{self.uuid} extract_entity_all cost time: {end_time - start_time} s"
        )

        start_time = time.time()
        strs = entity[:6]

        self.logger.info(f"{self.uuid} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]

        strs = prompt_preprocess(strs)

        self.logger.info(f"{self.uuid} {long_prompt} {description} {entity}")
        self.logger.info(f"{self.uuid} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4] != "":
            suit_enale = True

        gender = entity[-1]
        if gender not in genders:
            self.logger.info(f"{self.uuid} long_txt_retrive_npc gender error")
            return ujson.dumps([None])

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender + "_" + part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{self.uuid} key:{s} None_num:{None_num}")

            # if None_num==3 and (''!=s and ' '!=s  and None!=s and 'None'!=s):
            if None_num == 3 and key_id!=0 and key_id!=3:
                keys = retrive_single_txt(
                    description[:70], gender + "_" + part_keys[key_id]
                )

            self.logger.info(f"{self.uuid} keys:{keys}")
            for i in range(len(keys)):
                if keys[i] != None:
                    # print(keys)
                    img_paths_out.append(g_json_data[keys[i]]["Preview"])
                    glb_paths_out.append(g_json_data[keys[i]]["GLB_Mesh"])
                else:
                    img_paths_out.append(None)
                    glb_paths_out.append(None)
                keys_out.append(keys[i])

        final_out.append([glb_paths_out[0], glb_paths_out[1], glb_paths_out[2]])
        final_out.append([glb_paths_out[3], glb_paths_out[4], glb_paths_out[5]])
        final_out.append([glb_paths_out[6], glb_paths_out[7], glb_paths_out[8]])
        final_out.append([glb_paths_out[9], glb_paths_out[10], glb_paths_out[11]])
        final_out.append([glb_paths_out[12], glb_paths_out[13], glb_paths_out[14]])
        final_out.append([glb_paths_out[15], glb_paths_out[16], glb_paths_out[17]])

        final_out.append([keys_out[0], keys_out[1], keys_out[2]])
        final_out.append([keys_out[3], keys_out[4], keys_out[5]])
        final_out.append([keys_out[6], keys_out[7], keys_out[8]])
        final_out.append([keys_out[9], keys_out[10], keys_out[11]])
        final_out.append([keys_out[12], keys_out[13], keys_out[14]])
        final_out.append([keys_out[15], keys_out[16], keys_out[17]])

        final_out.append([img_paths_out[0], img_paths_out[1], img_paths_out[2]])
        final_out.append([img_paths_out[3], img_paths_out[4], img_paths_out[5]])
        final_out.append([img_paths_out[6], img_paths_out[7], img_paths_out[8]])
        final_out.append([img_paths_out[9], img_paths_out[10], img_paths_out[11]])
        final_out.append([img_paths_out[12], img_paths_out[13], img_paths_out[14]])
        final_out.append([img_paths_out[15], img_paths_out[16], img_paths_out[17]])
        final_out.append(entity[-1])  # gender
        final_out.append(suit_enale)
        if keys_out[0]!=None:
            final_out.append(entity[-2]) #hair color 20
        else:
            final_out.append(None)

        # list_bytes = pickle.dumps(final_out)
        # compressed_data = zlib.compress(list_bytes)

        # print(f"compressed_data:{get_size(compressed_data)}")
        # print(f"ori_data:{get_size(final_out)}")
        json_str = ujson.dumps(final_out)

        end_time = time.time()
        self.logger.info(
            f"{self.uuid} txt_retrive cost time: {end_time - start_time} s"
        )

        return json_str

    def exposed_long_retrive_npc_all_auto(self, long_prompt):
        """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output:glb path,fbx path,glb folder path
        """
        global g_json_data
        start_time = time.time()
        # strs = extract_entity(long_prompt).split('/')
        entity, description = extract_entity_all(long_prompt)

        end_time = time.time()
        self.logger.info(
            f"{self.uuid} extract_entity_all cost time: {end_time - start_time} s"
        )

        start_time = time.time()
        strs = entity[:6]

        self.logger.info(f"{self.uuid} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]

        strs = prompt_preprocess(strs)

        self.logger.info(f"{self.uuid} {long_prompt} {description} {entity}")
        self.logger.info(f"{self.uuid} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4] != "":
            suit_enale = True
        gender = entity[-1]

        if gender not in genders:
            self.logger.info(f"{self.uuid} long_retrive_npc_all_auto gender error")
            return ujson.dumps([None])

        body_shape = entity[-3]  ##body

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender + "_" + part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{self.uuid} key:{s} None_num:{None_num}")

            # if None_num==3 and (''!=s and ' '!=s  and None!=s and 'None'!=s):
            if None_num == 3:
                keys = retrive_single_txt(
                    description[:70], gender + "_" + part_keys[key_id]
                )

            self.logger.info(f"{self.uuid} keys:{keys}")
            for i in range(len(keys)):
                if keys[i] != None:
                    # print(keys)
                    img_paths_out.append(g_json_data[keys[i]]["Preview"])
                    glb_paths_out.append(g_json_data[keys[i]]["GLB_Mesh"])
                else:
                    img_paths_out.append(None)
                    glb_paths_out.append(None)
                keys_out.append(keys[i])

        if keys_out[0]!=None:
            hair_color = entity[-2] #hair color 20
        else:
            hair_color = None

        hair_path = glb_paths_out[0]
        top_path = glb_paths_out[3]
        bottom_path = glb_paths_out[6]
        shoe_path = glb_paths_out[9]
        outfit_path = glb_paths_out[12]
        others_path = glb_paths_out[15]

        hair_key = keys_out[0]
        top_key = keys_out[3]
        bottom_key = keys_out[6]
        shoe_key = keys_out[9]
        outfit_key = keys_out[12]
        others_key = keys_out[15]

        if suit_enale == False:
            path_list = [hair_path, top_path, bottom_path, shoe_path, None, others_path]
            key_list = [hair_key, top_key, bottom_key, shoe_key, None, others_key]
        else:
            path_list = [hair_path, None, None, shoe_path, outfit_path, others_path]
            key_list = [hair_key, None, None, shoe_key, outfit_key, others_key]

        texture_replace = [False, False, False, False, False, False]
        paths_temp = {}
        attr_keys = ["hair", "top", "trousers", "shoe", "outfit", "others"]
        for i, key in enumerate(key_list):
            if key != None and key != "" and key != " " and texture_replace[i] == False:
                paths_temp[g_json_data[key]["Obj_Mesh"]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }
            elif (
                key != None and key != "" and key != " " and texture_replace[i] == True
            ):
                paths_temp[path_list[i]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }

        # {"path":{"path1":{'cat':**,'key':**},"path2":attr2},"body_attr":[str1,str2]}

        paths_temp_in = {
            "path": paths_temp,
            "body_attr": [gender, body_shape],
            "hair_color": hair_color,
        }
        self.logger.info(f"{self.uuid} paths_temp_in:{paths_temp_in}")

        if self.uuid != None:
            mesh_output_path = model_save_folder+ str(self.uuid)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+ str(unique_id)
        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)

        start_time = time.time()
        self.wrap_cloth(mesh_output_path, paths_temp_in)

        end_time = time.time()
        self.logger.info(f"{self.uuid} wrap_cloth cost time: {end_time - start_time} s")

        start_time = time.time()
        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} app_autoRig_layer combine Response:{res}")
        else:
            self.logger.error(
                f"{self.uuid} app_autoRig_layer combine Request failed with status code"
            )
            return None, None, []

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{self.uuid} app_autoRig_layer combine Request failed with status code"
            )
            return None, None, []

        end_time = time.time()
        print("combine cost time: {:.2f} s".format(end_time - start_time))

        start_time = time.time()
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} autoRig_layer auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{self.uuid} autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []
        # auto_rig_layer(mesh_output_path)

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.fbx")):
            self.logger.error(
                f"{self.uuid} autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []

        end_time = time.time()
        self.logger.info(f"{self.uuid} auto_rig cost time: {end_time - start_time} s")

        start_time = time.time()
        self.logger.info(f"{self.uuid} mesh_output_path:{mesh_output_path}")

        fbx_path = os.path.join(mesh_output_path, "mesh/mesh.fbx")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        try:
            animation_gif(fbx_path, gif_path, self.logger)
        except:
            self.logger.error(f"{self.uuid} animation error")
            return None, None, []

        end_time = time.time()
        self.logger.info(f"{self.uuid} animation cost time: {end_time - start_time} s")

        final_out = [
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            os.path.join(mesh_output_path, "mesh/mesh_animation.fbx"),
            gif_path,
        ]
        json_str = ujson.dumps(final_out)

        return json_str

    def exposed_retrive_npc_all_auto(
        self,
        path_list,
        key_list,
        texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color="",
    ):
        """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output:glb path,fbx path,glb folder path
        """

        if (key_list[1] == "" or key_list[2] == "") and key_list[4] == "":
            self.logger.error(f"{self.uuid} retrive_npc_all_auto:path_list not correct")
            return None, None, []

        global g_json_data
        paths_temp = {}
        attr_keys = ["hair", "top", "trousers", "shoe", "outfit", "others"]
        for i, key in enumerate(key_list):
            if key != None and key != "" and key != " " and texture_replace[i] == False:
                paths_temp[g_json_data[key]["Obj_Mesh"]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }
            elif (
                key != None and key != "" and key != " " and texture_replace[i] == True
            ):
                paths_temp[path_list[i]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }

        # {"path":{"path1":{'cat':**,'key':**},"path2":attr2},"body_attr":[str1,str2]}

        paths_temp_in = {
            "path": paths_temp,
            "body_attr": [gender, shape],
            "hair_color": hair_color,
        }
        self.logger.info(f"{self.uuid} paths_temp_in:{paths_temp_in}")

        if self.uuid != None:
            mesh_output_path = model_save_folder+ str(self.uuid)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+ str(unique_id)
        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)

        start_time = time.time()
        self.wrap_cloth(mesh_output_path, paths_temp_in)

        end_time = time.time()
        self.logger.info(f"{self.uuid} wrap_cloth cost time: {end_time - start_time} s")

        start_time = time.time()
        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} app_autoRig_layer combine Response:{res}")
        else:
            self.logger.erroe(
                f"{self.uuid} app_autoRig_layer combine Request failed with status code"
            )
            return None, None, []

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{self.uuid} app_autoRig_layer combine Request failed with status code"
            )
            return None, None, []

        end_time = time.time()
        self.logger.info(f"{self.uuid} combine cost time: {end_time - start_time} s")

        start_time = time.time()
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} autoRig_layer auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{self.uuid} autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.fbx")):
            self.logger.error(
                f"{self.uuid} autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []

        end_time = time.time()
        self.logger.info(f"{self.uuid} auto_rig cost time: {end_time - start_time} s")

        start_time = time.time()
        self.logger.info(f"{self.uuid} mesh_output_path:{mesh_output_path}")

        fbx_path = os.path.join(mesh_output_path, "mesh/mesh.fbx")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        try:
            animation(fbx_path, gif_path, self.logger)
        except:
            self.logger.error(f"{self.uuid} animation error")
            return None, None, []
        end_time = time.time()
        self.logger.info(f"{self.uuid} animation cost time: {end_time - start_time} s")

        return (
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            gif_path,
            [mesh_output_path],
        )

    def exposed_retrive_npc_dislike(self, mesh_output_path):
        """
        retrive_npc_dislike
        input:txt example('/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/0a58f6f2-c40b-5e46-8f93-302e79f3caf0')
        output:None

        """
        templist = []
        templist.append(mesh_output_path)
        if len(templist) == 0:
            return
        print(templist)
        f = open(
            "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/dislike.txt",
            "a+",
        )
        f.write(templist[0])
        f.write("\n")
        f.close()

    def exposed_retrive_npc_animation_dislike(self, mesh_output_path):
        """
        retrive_npc_dislike
        input:txt example('/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/0a58f6f2-c40b-5e46-8f93-302e79f3caf0')
        output:None

        """
        templist = []
        templist.append(mesh_output_path)
        if len(templist) == 0:
            return
        print(templist)
        f = open(
            "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/animation_dislike.txt",
            "a+",
        )
        f.write(templist[0])
        f.write("\n")
        f.close()

    def exposed_retrive_npc_auto_binding_animation(self, mesh_output_path):
        """
        retrive_npc_auto_binding_animation
        input:mesh_output_path
        output:glb path,gif path,glb folder path
        """

        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}

        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} autoRig_layer auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{self.uuid} autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.fbx")):
            self.logger.error(
                f"{self.uuid} autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []

        self.logger.info(f"{self.uuid} mesh_output_path:{mesh_output_path}")
 
        fbx_path = os.path.join(mesh_output_path, "mesh/mesh.fbx")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        try:
            animation(fbx_path, gif_path, self.logger)
        except:
            self.logger.error(f"{self.uuid} animation error")
            return None, None, []

        return (
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            gif_path,
            [mesh_output_path],
        )

    def exposed_retrive_npc_combine(
        self,
        path_list,
        key_list,
        texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color="gold",
    ):
        """
        retrive_npc_manual_binding
        input:txt example(['/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Female hair/Female hair/F_HAIR_346/F_HAIR_346_fbx2020.glb',None, '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Bottoms/Bottoms/Bottoms01/BTM_93/BTM_93_fbx2020.glb', '/mnt/business_1/Data/DesignCenter/clock_fix_sample/20231215/fix_top_bottom/component/shoe/SK_Shoe_Sneaker03_F/SK_Shoe_Sneaker03_F.glb','/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/modify/Dresses/F_A/DR_673_F_A/DR_673_fbx2020.glb',  '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Glove Socks/Glove Socks/Socks/SOX_129/SOX_129_fbx2020.glb'],[F_HAIR_346_Asset,None,DSBA_BTM_4_Bottoms03,SH_241_SHOES01,DR_673_F_A_Dresses,SOX_204_SOX])
        if not be selected,that shoud be None
        output:glb folder path,glb path

        """
        if (key_list[1] == "" or key_list[2] == "") and key_list[4] == "":
            self.logger.error(f"{self.uuid} retrive_npc_combine:path_list not correct")
            return ujson.dumps([None, None])

        paths_temp = {}
        attr_keys = ["hair", "top", "trousers", "shoe", "outfit", "others"]
        self.logger.info(
            f"{self.uuid} path_list:{path_list}  key_list:{key_list}  texture_replace:{texture_replace}"
        )
        global g_json_data
        for i, key in enumerate(key_list):
            if key != None and key != "" and key != " " and texture_replace[i] == False:
                paths_temp[g_json_data[key]["Obj_Mesh"]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }
            elif (
                key != None and key != "" and key != " " and texture_replace[i] == True
            ):
                paths_temp[path_list[i]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }

        paths_temp_in = {
            "path": paths_temp,
            "body_attr": [gender, shape],
            "hair_color": hair_color,
        }
        self.logger.info(f"{self.uuid} paths_temp_in:{paths_temp_in}")

        if self.uuid != None:
            mesh_output_path = model_save_folder+str(self.uuid)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+ str(unique_id)
        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)

        start_time = time.time()
        self.wrap_cloth(mesh_output_path, paths_temp_in)
        end_time = time.time()
        self.logger.info(f"{self.uuid} wrap_cloth cost time: {end_time - start_time} s")

        self.logger.info(f"{self.uuid} mesh_output_path:{mesh_output_path}")

        start_time = time.time()
        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} app_autoRig_layer combine Response:{res}")
        else:
            self.logger.error(
                f"{self.uuid} app_autoRig_layer combine Request failed with status code"
            )
            return ujson.dumps([None, None])

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.fbx")):
            self.logger.error(
                f"{self.uuid} app_autoRig_layer combine Request failed with status code"
            )
            return ujson.dumps([None, None])

        json_str = ujson.dumps(
            [mesh_output_path, os.path.join(mesh_output_path, "mesh/mesh.glb")]
        )
        end_time = time.time()
        self.logger.info(f"{self.uuid} combine cost time: {end_time - start_time} s")
        return json_str


    def exposed_retrive_npc_autoRig(self, mesh_output_path):
        """
        retrive_npc_autoRig
        input:mesh folder path
        output:mesh folder path,glb path
        """
        self.logger.info(f"{self.uuid} mesh_output_path:{mesh_output_path}")
        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}

        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} autoRig_layer auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{self.uuid}autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.fbx")):
            self.logger.error(
                f"{self.uuid} autoRig_layer auto_rig Request failed with status code"
            )
            return None, None, []

        return mesh_output_path, os.path.join(mesh_output_path, "mesh/mesh.fbx")

    def exposed_retrive_npc_manual_binding(self, mesh_output_path):
        """
        retrive_npc_manual_binding
        input:mesh folder path
        output:mesh folder path

        todo
        """

        return mesh_output_path

    def exposed_retrive_npc_animation(self, mesh_output_path):
        """
        retrive_npc_animation
        input:glb folder path
        output:glb path,gif path,glb folder path

        """

        fbx_path = os.path.join(mesh_output_path, "mesh/mesh.fbx")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        self.logger.info(f"{self.uuid} mesh_output_path:{mesh_output_path}")
        try:
            animation(fbx_path, gif_path, self.logger)
        except:
            self.logger.error(f"{self.uuid} animation error")
            return None, None, []

        return (
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            gif_path,
            [mesh_output_path],
        )

    def exposed_retrive_npc_text_animation(self, mesh_output_path, text_prompt):
        """
        retrive_npc_animation
        input:glb folder path,text_prompt
        output:glb path,gif path,glb folder path
        """

        fbx_path = os.path.join(mesh_output_path, "mesh/mesh.fbx")
        self.logger.info(
            f"{self.uuid} mesh_output_path:{mesh_output_path} text_prompt:{text_prompt}"
        )
        try:
            animation_text(fbx_path, text_prompt, self.logger)
        except:
            self.logger.error(f"{self.uuid} text animation error")
            return None, None, []

        final_out = os.path.join(mesh_output_path, "mesh/mesh_animation.fbx")
        json_str = ujson.dumps(final_out)
        return json_str

    def exposed_auto_rig_manual_render(self, mesh_out_path):
        """
        npc_manual_binding
        input:mesh folder path
        outputLimage & joints_json
        """
        input_path = os.path.dirname(mesh_out_path)
        input = {"obj_path": input_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} app_autoRig auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{self.uuid} app_autoRig auto_rig failed with status code"
            )
            return None

        bone_pic = os.path.join(input_path, "show_image.png")
        jointsjson_path = os.path.join(input_path, "joints.json")
        return (bone_pic, jointsjson_path)

    def exposed_auto_rig_manual_calculate(self, joints_data, mesh_out_path):
        """
        npc_manual_binding
        input:mesh folder path
        outputLimage & joints_json
        """
        input_path = os.path.dirname(mesh_out_path)
        input = {"obj_path": input_path, "key_pts": joints_data}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} app_autoRig auto_rig Response:{res}")
        else:
            self.logger.info(
                f"{self.uuid} app_autoRig auto_rig failed with status code"
            )
            return None
        model_path = os.path.join(input_path, "mesh.fbx")
        resp = {"model_path": model_path}
        return resp

    def exposed_stop(self, type=None):
        print("exposed_stop==========={}".format(type))
        msg = "停止成功:"
        if type == "generateObject":
            msg = msg + "generateObject"
        else:
            msg = "无效类型"
        return msg

    def exposed_available(self, type="generateObject"):
        print("exposed_available==========={}".format(type))
        available = True
        return available

    def exposed_generateObject_with_image(
        self,
        local_image_path,
        face_num=10000,
        save_folder="/aigc_cfs_2/xiaqiangdai/data/obj_generate_temp",
        url="0.0.0.0",
    ):
        self.logger.info(
            f"{self.uuid} exposed_generateObject_with_image====={local_image_path}======{face_num}===={url}"
        )
        result = []
        fininal_result = ""
        try:
            generaStartTime = time.time()
            client = Client(url)

            self.logger.info(
                f"{self.uuid} local_image_path size:{os.path.getsize(local_image_path)}"
            )

            result = client.predict(local_image_path, api_name="/generate_mesh")

            self.logger.info(f"{self.uuid} 生成模型结果:{result}")
            generaEndTime = time.time()
            self.logger.info(
                f"{self.uuid} 生成模型花费时间:{generaEndTime-generaStartTime}"
            )
            copyStartTime = time.time()
            result = result[1]
            unique_id = result.split("/")[3]
            if not os.path.exists(os.path.join(save_folder, unique_id)):
                cmd1 = "mkdir " + os.path.join(save_folder, unique_id)
                self.logger.info(f"{self.uuid} {cmd1}")
                os.system(cmd1)
            cmd2 = "cp " + result + " " + os.path.join(save_folder, unique_id)
            self.logger.info(f"{self.uuid} {cmd2}")
            os.system(cmd2)
            copyEndTime = time.time()
            self.logger.info(
                f"{self.uuid} 拷贝模型花费时间:{copyEndTime-copyStartTime}"
            )
            # fininal_result=(str(result),)
            fininal_result = os.path.join(save_folder, unique_id + "/mesh.glb")
            self.logger.info(f"{self.uuid} fininal_result:{fininal_result}")
        except Exception as ex:
            self.logger.error(f"{self.uuid} 调用生成物体出错了:{ex}")
        return fininal_result

    def exposed_init_chat(self):
        self.logger.info(f"{self.uuid} exposed_init_chat begin")
        from chat_service import ChatService

        # chatservice = ChatService()
        # ret = chatservice.request_chat(0.2)
        # return chatservice, ret["data"]["response"]
        try:
            chatservice = ChatService()
            ret = chatservice.request_chat(0.2)
            return chatservice, ret["data"]["response"]
        except:
            self.logger.error(f"{self.uuid} exposed_init_chat fail")
            return None

    def exposed_generateRole_with_chat(self, chatservice, text, img):

        self.logger.info(f"{self.uuid} exposed_generateRole_with_chat begin")

        chatservice.update_chat_history(text, img)
        try:
            ret = chatservice.request_summary()
        except:  # 总结失败
            self.logger.error(f"{self.uuid} exposed_generateRole_with_chat summary fail")
            return None

            #  怎么处理？

        chatservice.update_history(text, img)

        # 请求下一轮对话成功
        try:
            ret = chatservice.request_chat(0.7)
            return ret["data"]["response"]
        except:
            self.logger.error(f"{self.uuid} exposed_generateRole_with_chat chat fail")
            return None

    def exposed_wework_ibot_retrive_npc(self,long_prompt):
        """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output:glb path,fbx path,glb folder path
        """
        global g_json_data
        start_time = time.time()
        # strs = extract_entity(long_prompt).split('/')
        entity, description = extract_entity_all(long_prompt)

        end_time = time.time()
        self.logger.info(f"{self.uuid} extract_entity_all cost time: {end_time - start_time} s")

        start_time = time.time()
        strs = entity[:6]

        self.logger.info(f"{self.uuid} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]
        
        strs = prompt_preprocess(strs)

        self.logger.info(f"{self.uuid} {long_prompt} {description} {entity}")
        self.logger.info(f"{self.uuid} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4]!='':
            suit_enale = True
        gender = entity[-1]
        hair_color = entity[-2]
        body_shape = entity[-3]

        if gender not in genders:
            self.logger.info(f"{self.uuid} wework_ibot_retrive_npc gender error")
            return ujson.dumps([None])

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender+'_'+part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{self.uuid} key:{s} None_num:{None_num}")

            # if None_num==3 and (''!=s and ' '!=s  and None!=s and 'None'!=s):
            if None_num == 3:
                keys = retrive_single_txt(description[:70], gender+'_'+part_keys[key_id])

            self.logger.info(f"{self.uuid} keys:{keys}")
            for i in range(len(keys)):
                if keys[i] != None:
                    # print(keys)
                    img_paths_out.append(g_json_data[keys[i]]["Preview"])
                    glb_paths_out.append(g_json_data[keys[i]]["GLB_Mesh"])
                else:
                    img_paths_out.append(None)
                    glb_paths_out.append(None)
                keys_out.append(keys[i])

        hair_path = glb_paths_out[0]
        top_path = glb_paths_out[3]
        bottom_path = glb_paths_out[6]
        shoe_path = glb_paths_out[9]
        outfit_path = glb_paths_out[12]
        others_path = glb_paths_out[15]

        hair_key = keys_out[0]
        top_key = keys_out[3]
        bottom_key = keys_out[6]
        shoe_key = keys_out[9]
        outfit_key = keys_out[12]
        others_key = keys_out[15]

        if suit_enale==False:
            path_list = [hair_path,top_path,bottom_path,shoe_path,None,others_path]
            key_list = [hair_key,top_key,bottom_key,shoe_key,None,others_key]
        else:
            path_list = [hair_path,None,None,shoe_path,outfit_path,others_path]
            key_list = [hair_key,None,None,shoe_key,outfit_key,others_key]


        texture_replace=[False, False, False, False, False, False]
        paths_temp = {}
        attr_keys = ["hair", "top", "trousers", "shoe", "outfit", "others"]
        for i, key in enumerate(key_list):
            if key != None and key != "" and key != " " and texture_replace[i] == False:
                paths_temp[g_json_data[key]["Obj_Mesh"]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }
            elif (
                key != None and key != "" and key != " " and texture_replace[i] == True
            ):
                paths_temp[path_list[i]] = {
                    "cat": attr_keys[i],
                    "asset_key": key,
                    "key": g_json_data[key]["body_key"],
                }

        # {"path":{"path1":{'cat':**,'key':**},"path2":attr2},"body_attr":[str1,str2]}
        shape="fat"
        paths_temp_in = {"path": paths_temp, "body_attr": [gender, body_shape],"hair_color":hair_color}
        self.logger.info(f"{self.uuid} paths_temp_in:{paths_temp_in}")

        if self.uuid!=None:
            mesh_output_path = model_save_folder+ str(self.uuid)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+str(unique_id)
        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)

        start_time = time.time()
        self.wrap_cloth(mesh_output_path, paths_temp_in)

        end_time = time.time()
        self.logger.info(f"{self.uuid} wrap_cloth cost time: {end_time - start_time} s")

        start_time = time.time()
        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}
        res = requests.post(
            "",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{self.uuid} app_autoRig_layer combine Response:{res}")
        else:
            self.logger.error(f"{self.uuid} app_autoRig_layer combine Request failed with status code")
            return None, None, []
        
        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.fbx")):
            self.logger.error(f"{self.uuid} app_autoRig_layer combine Request failed with status code")
            return None, None, []

        end_time = time.time()
        self.logger.info(f"{self.uuid} combine cost time: {end_time - start_time} s")

        final_out= [os.path.join(mesh_output_path, "mesh/mesh.glb"), path_list, key_list, gender, shape]
        json_str = ujson.dumps(final_out)

        return json_str
    
    def exposed_wework_ibot_animation(self, mesh_output_path):
        start_time = time.time()
        self.logger.info(f"{self.uuid}  mesh_output_path:{mesh_output_path}")

        fbx_path = os.path.join(mesh_output_path, "mesh/mesh.fbx")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        try:
            animation_gif(fbx_path,gif_path,self.logger)
        except:
            self.logger.error(f"{self.uuid} gif animation error")
            return None, None, []
        end_time = time.time()
        self.logger.info(f"{self.uuid} animation cost time: {end_time - start_time} s")

        final_out= [os.path.join(mesh_output_path, "mesh/mesh_animation.fbx"),gif_path]
        json_str = ujson.dumps(final_out)

        return json_str

    def exposed_render_gif_frontImage(self, model_path):
        """
        render_gif_frontImage
        input:model_path
        output:360 video and front image

        """
        self.logger.info(f"{self.uuid}  model_path:{model_path}")
        model_folder = os.path.dirname(model_path)
        render_foler  = os.path.join(model_folder,'render')
        if not os.path.exists(render_foler):
            os.mkdir(render_foler)
    
        cmd1 = f"/root/blender-3.6.5-linux-x64/blender -b -P \
        /aigc_cfs_2/xiaqiangdai/project/render_gif_frontImage/render_color_depth_normal_helper.py -- \
        --mesh_path {model_path}  \
        --output_folder {render_foler} \
        --pose_json_path /mnt/aigc_cfs_cq/xiaqiangdai/project/render_gif_frontImage/20240407-S-120-cam_parameters.json  \
        --white_background  --no_solidify --save_gif  --engine eevee --render_height 256 --render_width 256 --only_render_png  --smooth "
        os.system(cmd1)
        os.system("rm "+render_foler+"/color/*png")
        cmd2 = f"/root/blender-3.6.5-linux-x64/blender -b -P \
        /aigc_cfs_2/xiaqiangdai/project/render_gif_frontImage/render_color_depth_normal_helper.py -- \
        --mesh_path {model_path}  \
        --output_folder {render_foler} \
        --pose_json_path /mnt/aigc_cfs_cq/xiaqiangdai/project/render_gif_frontImage/20240328-S-1-cam_parameters.json  \
        --white_background  --no_solidify  --engine eevee --render_height 256 --render_width 256 --only_render_png  --smooth "
        os.system(cmd2)
        final_out = [os.path.join(render_foler,'color/cam-0000.png'),os.path.join(render_foler,'color/render.webm')]
        json_str = ujson.dumps(final_out)

        return json_str
    
    def exposed_render_shader(self, model_path):
        """
        render_shader
        input:model_path
        output:shader video and front image
        """
        self.logger.info(f"{self.uuid}  render_shader model_path:{model_path}")
        model_folder = os.path.dirname(model_path)
        render_foler  = os.path.join(model_folder,'render')
        blenderFile_path = "/aigc_cfs_2/xiaqiangdai/project/server_bakend_temp/base_interface/shader/RenderFactory.blend"
        RenderMesh(blenderFile_path,model_path)
        render2video(model_path)
        if not os.path.exists(os.path.join(render_foler,'CoverCollect.png')) or not os.path.exists(os.path.join(render_foler,'render.webm')):
            self.logger.error(f"{self.uuid} render_shader model_path:{model_path} error")
            return ujson.dumps([None,None])
        final_out = [os.path.join(render_foler,'CoverCollect.png'),os.path.join(render_foler,'render.webm')]
        json_str = ujson.dumps(final_out)

        return json_str
    
    def exposed_hair_is_change_ok(self,mesh_path):
        """
        hair_is_change_ok
        input:mesh_path
        output:True or False
        """
        try:
            ret = is_change_ok(mesh_path)
        except Exception as e:
            self.logger.error(f"{self.uuid} hair_is_change_ok error")
            return False

        return ret
    
    def exposed_get_retrieve_json_path(self):
        """
        hair_is_change_ok
        input:
        output:cfs and gdp json path
        """
        final_out=[json_path,gdp_json_path]
        json_str = ujson.dumps(final_out)
        return json_str



if __name__ == "__main__":
    # while True:
    args = parser.parse_args()
    print("启动所有RPC:{}".format(args))

    server1 = ThreadPoolServer(
        RPCService, hostname=args.hostname1, port=80, nbThreads=128
    )
    # server1 = ThreadPoolServer(
    #     RPCService, hostname=args.hostname1, port=8083, protocol_config=rpyc_config
    # )
    server1.start()
