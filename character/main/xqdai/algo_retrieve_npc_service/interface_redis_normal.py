from gradio_client import Client
import sys
import os,shutil
from flask import Flask, jsonify
import requests

app = Flask(__name__)

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.insert(0, parent_dir)

sys.path.append(os.path.join(parent_dir,"npc_layer_retrieve/retrieve"))
sys.path.append(os.path.join(parent_dir,'animation'))
sys.path.append(os.path.join(parent_dir,"auto_rig/layer/change_hair"))
sys.path.append(os.path.join(parent_dir,"auto_rig/layer/tdmq_interface"))
sys.path.append(os.path.join(parent_dir,"cloth_wrap/webui"))
from cos import CosClient
from single_retrieve_normic_test import retrive_single_txt
import uuid
import time
import json
import requests
from retrieval_normic import keys_dict

from shape_retrival_rerank import shape_text_retrieve
from base_body_map import base_body_map

import argparse
import threading
import zlib
import pickle
from extract_entity import extract_entity_all
import ujson
import logging
from datetime import datetime
from animation_interface import animation, animation_text, animation_text_retrieve,animation_text_retrieve_retarget
from distribute_logging import *
from shader.Run_Render import *
from is_change_ok import is_change_ok_online
import redis
from easy_interface import CombineEasyInterface
from clothwrap2.tdmq_interface.easy_interface  import clothwrape_EasyInterface
from tdmq_everything.gputool.tdmq_client_gputool import GpuToolInterface, init_job_id

current_file_path = os.path.abspath(__file__)
current_file_directory = os.path.dirname(current_file_path)

json_path = "/aigc_cfs_gdp/xiaqiangdai/retrieve_libs//20241010_daz_decimate_add_ct.json"
gdp_json_path = (
    "/aigc_cfs_gdp/xiaqiangdai/retrieve_libs//20241010_daz_decimate_add_ct.json"
)

model_save_folder = "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/"
cos_model_save_folder = "/mnt/aigc_bucket_4/pandorax/retrieveNPC_save/"
log_folder = "/aigc_cfs_gdp/xiaqiangdai/logs/retrieveNPC_main"

global g_json_data
g_json_data = keys_dict(json_path)


@app.route('/health', methods=['GET'])
def health_check():
    
    health_status = {
        'status': 'UP',
        'message': 'Service is healthy'
    }
    return jsonify(health_status), 200


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
    current_time = datetime.now()
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(
                os.path.getctime(file_path)
            )
            file_age = (current_time - file_creation_time).days
            if file_age > days:
                os.remove(file_path)
                print(f"Removed file: {file_path}")

def move_files_older_than_n_days(directory,dst_directory, days):
    current_time = datetime.now()
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(
                os.path.getctime(file_path)
            )
            file_age = (current_time - file_creation_time).days
            if file_age > days:
                shutil.move(file_path, dst_directory)
                print(f"move file: {file_path} to {dst_directory}")

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


def connect_redis(host="",password="",port=6379,db_id=0):
    while True:
        try:
            r = redis.Redis(
                host=host,
                password=password,
                port=port,
                db=db_id,
                health_check_interval=30,
            )
            # r = redis.Redis(host='localhost', port=6379, db=0,health_check_interval=30)
            r.ping()
            return r
        except redis.exceptions.ConnectionError as e:
            print("连接失败，尝试重新连接...")
            time.sleep(5)


class RetrieveNPCService():

    def __init__(self,host="",password='',port=6379,db_id=0,scope='webui'):

        self.host= host
        self.password= password
        self.port= port
        self.db_id= db_id
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day

        self.log_file_path = (
            f"{log_folder}/{year}_{month}_{day}.log"
        )
        self.logger = DistributedFileLogger("RetrieveNPCService_logger", self.log_file_path)
        self.sleepTime = 10
        self.processNum = 0

        # self.redis_conn = redis.Redis(host='localhost', password='TX_123',port=6379, db=0)
        self.redis_conn = connect_redis(host=self.host,password=self.password,port=self.port,db_id=self.db_id)

        self.redis_conn.delete("npc_retrieve_redis_lock")
        # self.redis_conn.delete("message_status")
                   
        self.num_threads = 50
        self.queue_threads = []
        for i in range(self.num_threads):
            queue_thread = threading.Thread(target=self.process_queue)
            queue_thread.daemon = True
            queue_thread.start()
            self.queue_threads.append(queue_thread)
        self.queue_length = 0
        update_thread = threading.Thread(target=self.update)
        update_thread.daemon = True
        update_thread.start()
        print(f"npc_retrieve_queue:{self.redis_conn.llen('npc_retrieve_queue')}")
        self.messages = [
            "cos_upload",
            "cos_upload_remote",
            "texReplace",
            "txt_retrive_npc",
            "long_txt_retrive_npc",
            "long_retrive_npc_all_auto",
            "retrive_npc_all_auto",
            "retrive_npc_dislike",
            "retrive_npc_animation_dislike",
            "retrive_npc_auto_binding_animation",
            "retrive_npc_combine",
            "retrive_npc_autoRig",
            "retrive_npc_manual_binding",
            "retrive_npc_animation",
            "retrive_npc_text_animation",
            "npc_animation_text_retrieve",
            "animation_gif_generate",
            "render_gif_frontImage",
            "render_shader",
            "shape_text_retrieve",
            "hair_is_change_ok",
            "get_retrieve_json_path",
        ]
        self.messages_num = {}

        client_cfg_json = os.path.join(current_file_directory,"auto_rig/layer/tdmq_interface/configs/tdmq_combine.json")
        self.combine_interface = CombineEasyInterface(client_cfg_json)
        clothwrape_client_cfg_json =os.path.join(current_file_directory,"clothwrap2/tdmq_interface/configs/tdmq.json")
        self.clothwrape_interface = clothwrape_EasyInterface(clothwrape_client_cfg_json)
        blender_cvt_client_cfg_json =os.path.join(current_file_directory,'tdmq_everything/gputool/configs/tdmq_gputool.json')
        self.blender_cvt_interface = GpuToolInterface(blender_cvt_client_cfg_json)

    def update(self):
        time.sleep(3600*12)
        try:
            remove_files_older_than_n_days(log_folder, 10)
            move_files_older_than_n_days(model_save_folder,cos_model_save_folder,1)
        except Exception as e:
            print(e)
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day
        self.log_file_path = (
            f"{log_folder}/{year}_{month}_{day}.log"
        )
        self.logger = DistributedFileLogger("RetrieveNPCService_logger", self.log_file_path)


    def wrap_cloth(self, mesh_output_path:str, paths_temp:str,job_id:str):

        # paths_temp_in = {"path": paths_temp, "body_attr": [gender, shape],"hair_color":entity[0]}

        gender = paths_temp["body_attr"][0]
        shape_promt = paths_temp["body_attr"][1]
        try:
            body_attr = shape_text_retrieve(gender, shape_promt)
        except Exception as e:
            print(e)
            self.logger.error(f"{job_id} shape_text_retrieve exception")

        paths_temp["body_attr"] = body_attr
        paths_temp["shape_promt"] = shape_promt

        uuid_temp = str(uuid.uuid4())
        self.logger.info(f"{job_id} wrap_cloth {paths_temp}")
        obj_lst = os.path.join(mesh_output_path, f"object_lst_{uuid_temp}.txt")

        json_object = json.dumps(paths_temp, indent=4)

        self.logger.info(f"{job_id} json_object:{json_object}")
        with open(obj_lst, "w") as f:
            f.write(json_object)

        success_flag, result_meshs = self.clothwrape_interface.blocking_call(uuid_temp,obj_lst)
        if success_flag:
            return mesh_output_path
        else:
            return None

    # massage
    # channel:type_in
    # data:param_in
    def process_queue(self):
        
        while True:

            if self.redis_conn == None:
                self.redis_conn = connect_redis(host=self.host,password=self.password,port=self.port,db_id=self.db_id)

            self.redis_lock_acquired = self.redis_conn.setnx(
                "npc_retrieve_redis_lock", "locked"
            )
            if self.redis_lock_acquired:

                self.queue_length = self.redis_conn.llen("npc_retrieve_queue")
                if self.queue_length == 0:
                    self.redis_conn.delete("npc_retrieve_redis_lock")
                    continue
                else:
                    print(f"self.queue_length:{self.queue_length}")
                    message_str = self.redis_conn.rpop("npc_retrieve_queue")
                    self.redis_conn.delete("npc_retrieve_redis_lock")
            else:
                continue

            if message_str:
                message = json.loads(message_str)
                print(message)

                if message["channel"] not in self.messages:
                    self.logger.error(f"{message['channel']} not support")
                    continue
                print("message:", message)
     
                self.redis_conn.hset('message_status', '_'.join([message['channel'],message['id']]), "processing")
                try:
                    if message["channel"] == "cos_upload":
                        success, result = self.exposed_cos_upload(*message["data"])
                    if message["channel"] == "cos_upload_remote":
                        success, result = self.exposed_cos_upload_remote(*message["data"])
                    if message["channel"] == "txt_retrive_npc":
                        success, result = self.exposed_txt_retrive_npc(*message["data"])
                    if message["channel"] == "long_txt_retrive_npc":
                        success, result = self.exposed_long_txt_retrive_npc(
                            *message["data"]
                        )
                    if message["channel"] == "long_retrive_npc_all_auto":
                        success, result = self.exposed_long_retrive_npc_all_auto(
                            *message["data"]
                        )
                    if message["channel"] == "retrive_npc_all_auto":
                        success, result = self.exposed_retrive_npc_all_auto(
                            *message["data"]
                        )
                    if message["channel"] == "retrive_npc_dislike":
                        success, result = self.exposed_retrive_npc_dislike(*message["data"])
                    if message["channel"] == "retrive_npc_animation_dislike":
                        success, result = self.exposed_retrive_npc_animation_dislike(
                            *message["data"]
                        )
                    if message["channel"] == "retrive_npc_auto_binding_animation":
                        success, result = self.exposed_retrive_npc_auto_binding_animation(
                            *message["data"]
                        )
                    if message["channel"] == "retrive_npc_combine":
                        success, result = self.exposed_retrive_npc_combine(*message["data"])
                    if message["channel"] == "retrive_npc_autoRig":
                        success, result = self.exposed_retrive_npc_autoRig(*message["data"])
                    if message["channel"] == "retrive_npc_manual_binding":
                        success, result = self.exposed_retrive_npc_manual_binding(
                            *message["data"]
                        )
                    if message["channel"] == "retrive_npc_animation":
                        success, result = self.exposed_retrive_npc_animation(
                            *message["data"]
                        )
                    if message["channel"] == "retrive_npc_text_animation":
                        success, result = self.exposed_retrive_npc_text_animation(
                            *message["data"]
                        )
                    if message["channel"] == "npc_animation_text_retrieve":
                        success, result = self.exposed_npc_animation_text_retrieve(
                            *message["data"]
                        )
                    if message["channel"] == "animation_gif_generate":
                        success, result = self.exposed_animation_gif_generate(
                            *message["data"]
                        )
                    if message["channel"] == "render_gif_frontImage":
                        success, result = self.exposed_render_gif_frontImage(
                            *message["data"]
                        )
                    if message["channel"] == "render_shader":
                        success, result = self.exposed_render_shader(
                            *message["data"]
                        )
                    if message["channel"] == "shape_text_retrieve":
                        success, result = self.exposed_shape_text_retrieve(
                            *message["data"]
                        )
                    if message["channel"] == "hair_is_change_ok":
                        success, result = self.exposed_hair_is_change_ok(
                            *message["data"]
                        )
                    if message["channel"] == "get_retrieve_json_path":
                        success, result = self.exposed_get_retrieve_json_path(
                            *message["data"]
                        )
                    result = ujson.loads(result)
                except Exception as e:
                    print(e)
                    self.logger.error(f"{message['channel']} {message['id']} {e}")
                    success = False
                publish_channel =  "result_" + message["channel"] + "_" + str(message["id"])
                publish_channel = publish_channel.encode('utf-8')
                
                message_str = message["channel"] + "_" + str(message["id"])
                if message_str not in self.messages_num.keys():
                    self.messages_num[message_str]=0

                if not success and self.messages_num[message_str] < 5:
                    self.logger.info(
                        f"info:{'result_'+message_str} not sucess {self.messages_num[message_str]}"
                    )
                    self.redis_conn.lpush("npc_retrieve_queue", ujson.dumps(message))
                    self.messages_num[message_str] += 1
                    print("==================1111")
                    time.sleep(8)
                elif not success and self.messages_num[message_str] >= 5:
                    self.logger.error(
                        f"result_{message_str} not sucess {self.messages_num[message_str]}"
                    )
                    self.redis_conn.sadd('results', publish_channel)
                    self.redis_conn.set(publish_channel, json.dumps(
                            {
                                "method": message["channel"],
                                "args": [],
                                "result": "failed",
                            }
                        ))
                    
                    del self.messages_num[message_str]
                    self.redis_conn.hset('message_status', message_str, "failed")
                    print("==================222222")
                elif success:
                    self.logger.info(
                        f"success {'result_'+message_str}"
                    )
                    del self.messages_num[message_str]
                    self.redis_conn.hset('message_status', message_str, "finished")
                    for i in range(20):
                        status=''
                        try:
                            status = self.redis_conn.hget('message_status', message_str).decode()
                        except Exception as e:
                            pass
                        if status!='finished':
                            self.redis_conn.hset('message_status', message_str, "finished")
                            print(f"set status finished:{status}")
                        else:
                            break
                        time.sleep(0.1)
                    confirm_num = 0
                    while True:
                        confirm_num+=1

                        self.redis_conn.sadd('results', publish_channel)
                        self.redis_conn.set(publish_channel,
                            json.dumps(
                                {
                                    "method": message["channel"],
                                    "args": message["data"],
                                    "result": result,
                                }
                            )
                        )
                        print("==================3333333")
                        time.sleep(3)
                        status=''
                        try:
                            status = self.redis_conn.hget('message_status', message_str).decode()
                        except Exception as e:
                            pass
                        if status=='received':
                            # 收到确认消息
                            print('result received and confirmed')
                            break
                        else:
                            print(f"status:{status} {message_str}")
                            
                        if confirm_num>20:
                            self.logger.error(f"{message_str} message publish error {confirm_num}")
                            del self.messages_num[message_str]
                            # self.redis_conn.hset('message_status', '_'.join([message['channel'],message['id']]), "confirm_failed")
                            print("==================44444")
                            break
                   
                print(f"publish_channel:{publish_channel}")

    # cos_upload
    def exposed_cos_upload(self, glb_local_path,job_id):
        start_time = time.time()
        guid = uuid.uuid4().hex
        new_glb_name = f"{guid}/{os.path.basename(glb_local_path)}"
        cos_client = CosClient()
        remote_file_path = os.path.join(
            cos_client.retrieve_NPC_dir_remote, new_glb_name
        )
        glb_url = cos_client.upload(glb_local_path, remote_file_path)
        end_time = time.time()
        self.logger.info(f"{job_id} glb_url:{glb_url} cost time:{end_time - start_time}")
        json_str = ujson.dumps(glb_url)

        # self.redis_conn.set(f'result_cos_upload:{glb_local_path}', json_str)
        # self.redis_conn.publish('result_cos_upload', json.dumps({'method': 'cos_upload', 'args': [glb_local_path], 'result': glb_url}))

        return True, json_str

    def exposed_cos_upload_remote(self,glb_local_path,remote_file_path,job_id):
        try:
            start_time = time.time()
            cos_client = CosClient()
            file_local_path="/mnt/aigc_bucket_4" + remote_file_path
            # if os.path.exists(file_local_path):
            #     os.remove(file_local_path)
            glb_url = cos_client.upload(glb_local_path, remote_file_path)
            end_time = time.time()
            self.logger.info(f"cos_upload_remote {job_id} glb_local_path:{glb_local_path} remote_file_path:{remote_file_path} glb_url:{glb_url} cost time:{end_time - start_time}")
            json_str = ujson.dumps(glb_url)
            return True,json_str
        except Exception as e:
            self.logger.error(f"cos_upload_remote error {job_id} glb_local_path:{glb_local_path} remote_file_path:{remote_file_path}")
            return False,None


    # txt_retrive_npc
    def exposed_txt_retrive_npc(self, entity, description,job_id):
        """
        npc text retrieve
        input:txt list([hair,top,trousers,shoe,outfit,others,gender])
        output info:[hair image list,top image list,trousers image list,shoe image list,outfit image list,others image list,
                    hair glb list,top glb list,trousers glb list,shoe glb list,outfit glb list,others glb list,
                    hair key list,top key list,trousers key list,shoe key list,outfit key list,others key list]
        """

        start_time = time.time()
        strs = entity[:6]

        self.logger.info(f"{job_id} {description} {entity}")
        self.logger.info(f"{job_id} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]

        strs = prompt_preprocess(strs)

        self.logger.info(f"{job_id} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4] != "":
            suit_enale = True

        gender = entity[-1]
        if gender not in genders:
            self.logger.error(f"{job_id} txt_retrive_npc gender error")
            return False, ujson.dumps([None])

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender + "_" + part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{job_id} key:{s} None_num:{None_num}")

            if None_num == 3 and key_id!=0 and key_id!=3:
                keys = retrive_single_txt(
                    description[:70], gender + "_" + part_keys[key_id]
                )

            self.logger.info(f"{job_id} keys:{keys}")
            for i in range(len(keys)):
                if keys[i] != None:
                    # print(keys)
                    img_paths_out.append(g_json_data[keys[i]]["Preview"])
                    glb_paths_out.append(g_json_data[keys[i]]["GLB_Mesh"])
                else:
                    img_paths_out.append(None)
                    glb_paths_out.append(None)
                keys_out.append(keys[i])

        out_glb = [[glb_paths_out[0], glb_paths_out[1], glb_paths_out[2]],
                    [glb_paths_out[3], glb_paths_out[4], glb_paths_out[5]],
                    [glb_paths_out[6], glb_paths_out[7], glb_paths_out[8]],
                    [glb_paths_out[9], glb_paths_out[10], glb_paths_out[11]],
                    [glb_paths_out[12], glb_paths_out[13], glb_paths_out[14]],
                    [glb_paths_out[15], glb_paths_out[16], glb_paths_out[17]]]
        
        out_key = [[keys_out[0], keys_out[1], keys_out[2]],
                    [keys_out[3], keys_out[4], keys_out[5]],
                    [keys_out[6], keys_out[7], keys_out[8]],
                    [keys_out[9], keys_out[10], keys_out[11]],
                    [keys_out[12], keys_out[13], keys_out[14]],
                    [keys_out[15], keys_out[16], keys_out[17]]]
        
        out_img = [[img_paths_out[0], img_paths_out[1], img_paths_out[2]],
                    [img_paths_out[3], img_paths_out[4], img_paths_out[5]],
                    [img_paths_out[6], img_paths_out[7], img_paths_out[8]],
                    [img_paths_out[9], img_paths_out[10], img_paths_out[11]],
                    [img_paths_out[12], img_paths_out[13], img_paths_out[14]],
                    [img_paths_out[15], img_paths_out[16], img_paths_out[17]]]


        final_out={'model_path':out_glb,'key':out_key,'img':out_img,'gender':entity[-1],'suit_enale':suit_enale,'hair_color':entity[-2],'body':entity[-3],'action':entity[-4]}
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_txt_retrive_npc:{entity}:{description}', json_str)
        # self.redis_conn.publish('result_txt_retrive_npc', json.dumps({'method': 'txt_retrive_npc', 'args': [entity,description], 'result': final_out}))

        end_time = time.time()
        self.logger.info(
            f"{job_id} txt_retrive cost time: {end_time - start_time} s"
        )

        return True, json_str

    # long_txt_retrive_npc
    def exposed_long_txt_retrive_npc(self, long_prompt,job_id):
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
            f"{job_id} extract_entity_all cost time: {end_time - start_time} s"
        )

        start_time = time.time()
        strs = entity[:6]
        self.logger.info(f"{job_id} {long_prompt} {description} {entity}")
        self.logger.info(f"{job_id} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]

        strs = prompt_preprocess(strs)

        self.logger.info(f"{job_id} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4] != "":
            suit_enale = True

        gender = entity[-1]
        if gender not in genders:
            self.logger.info(f"{job_id} long_txt_retrive_npc gender error")
            return False, ujson.dumps([None])

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender + "_" + part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{job_id} key:{s} None_num:{None_num}")

            # if None_num==3 and (''!=s and ' '!=s  and None!=s and 'None'!=s):
            if None_num == 3 and key_id!=0 and key_id!=3:
                keys = retrive_single_txt(
                    description[:70], gender + "_" + part_keys[key_id]
                )

            self.logger.info(f"{job_id} keys:{keys}")
            for i in range(len(keys)):
                if keys[i] != None:
                    # print(keys)
                    img_paths_out.append(g_json_data[keys[i]]["Preview"])
                    glb_paths_out.append(g_json_data[keys[i]]["GLB_Mesh"])
                else:
                    img_paths_out.append(None)
                    glb_paths_out.append(None)
                keys_out.append(keys[i])

        out_glb = [[glb_paths_out[0], glb_paths_out[1], glb_paths_out[2]],
                    [glb_paths_out[3], glb_paths_out[4], glb_paths_out[5]],
                    [glb_paths_out[6], glb_paths_out[7], glb_paths_out[8]],
                    [glb_paths_out[9], glb_paths_out[10], glb_paths_out[11]],
                    [glb_paths_out[12], glb_paths_out[13], glb_paths_out[14]],
                    [glb_paths_out[15], glb_paths_out[16], glb_paths_out[17]]]
        
        out_key = [[keys_out[0], keys_out[1], keys_out[2]],
                    [keys_out[3], keys_out[4], keys_out[5]],
                    [keys_out[6], keys_out[7], keys_out[8]],
                    [keys_out[9], keys_out[10], keys_out[11]],
                    [keys_out[12], keys_out[13], keys_out[14]],
                    [keys_out[15], keys_out[16], keys_out[17]]]
        
        out_img = [[img_paths_out[0], img_paths_out[1], img_paths_out[2]],
                    [img_paths_out[3], img_paths_out[4], img_paths_out[5]],
                    [img_paths_out[6], img_paths_out[7], img_paths_out[8]],
                    [img_paths_out[9], img_paths_out[10], img_paths_out[11]],
                    [img_paths_out[12], img_paths_out[13], img_paths_out[14]],
                    [img_paths_out[15], img_paths_out[16], img_paths_out[17]]]


        final_out={'model_path':out_glb,'key':out_key,'img':out_img,'gender':entity[-1],'suit_enale':suit_enale,'hair_color':entity[-2],'body':entity[-3],'action':entity[-4]}
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_long_txt_retrive_npc:{long_prompt}', json_str)
        # self.redis_conn.publish('result_long_txt_retrive_npc', json.dumps({'method': 'long_txt_retrive_npc', 'args': [long_prompt], 'result': final_out}))

        end_time = time.time()
        self.logger.info(
            f"{job_id} txt_retrive cost time: {end_time - start_time} s"
        )

        return True, json_str

    # long_retrive_npc_all_auto
    def exposed_long_retrive_npc_all_auto(self, long_prompt,job_id):
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
            f"{job_id} extract_entity_all cost time: {end_time - start_time} s"
        )

        start_time = time.time()
        strs = entity[:6]

        self.logger.info(f"{job_id} {long_prompt} {description} {entity}")
        self.logger.info(f"{job_id} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]

        strs = prompt_preprocess(strs)

        self.logger.info(f"{job_id} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4] != "":
            suit_enale = True
        gender = entity[-1]

        if gender not in genders:
            self.logger.info(f"{job_id} long_retrive_npc_all_auto gender error")
            return False, ujson.dumps([None])

        hair_color = entity[-2]  ##hair color
        body_shape = entity[-3]  ##body

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender + "_" + part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{job_id} key:{s} None_num:{None_num}")

            # if None_num==3 and (''!=s and ' '!=s  and None!=s and 'None'!=s):
            if None_num == 3 and key_id!=0 and key_id!=3:
                keys = retrive_single_txt(
                    description[:70], gender + "_" + part_keys[key_id]
                )

            self.logger.info(f"{job_id} keys:{keys}")
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
        self.logger.info(f"{job_id} paths_temp_in:{paths_temp_in}")

        if job_id != None:
            mesh_output_path = model_save_folder+ str(job_id)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+ str(unique_id)
            
        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)

        start_time = time.time()
        wrap_cloth_result_flag = self.wrap_cloth(mesh_output_path, paths_temp_in,job_id)
        if wrap_cloth_result_flag==None:
            self.logger.error(f"{job_id} {mesh_output_path} wrap_cloth error")
            return False, [None, None, []]

        end_time = time.time()
        self.logger.info(f"{job_id} wrap_cloth cost time: {end_time - start_time} s")

        start_time = time.time()
        success_flag, result_meshs = self.combine_interface.blocking_call_combine(str(uuid.uuid4()),
                                                            mesh_output_path,
                                                            output_mesh_filename=os.path.join(mesh_output_path, "mesh/mesh.glb"))
        if success_flag:
            self.logger.info(f"{job_id} app_autoRig_layer combine Response")
        else:
            self.logger.error(
                f"{job_id} app_autoRig_layer combine Request failed with status code"
            )
            return False, [None, None, []]

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{job_id} app_autoRig_layer combine Request failed with status code"
            )
            return False, [None, None, []]

        end_time = time.time()
        print("combine cost time: {:.2f} s".format(end_time - start_time))

        start_time = time.time()
        success, out_gif = self.blender_cvt_interface.blocking_call_render_gif(init_job_id(), os.path.join(mesh_output_path, "mesh/mesh.glb"), os.path.join(mesh_output_path, "mesh/mesh.gif"))
        if success:
            self.logger.info(f"{job_id} blocking_call_render_gif success")
        else:
            self.logger.error(
                f"{job_id} blocking_call_render_gif failed"
            )
            return False, [None, None, []]
        
        end_time = time.time()
        print("blocking_call_render_gif cost time: {:.2f} s".format(end_time - start_time))

        start_time = time.time()
        res = requests.post(
            "http://url:8080/app_autoRig_layer/auto_rig",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{job_id} autoRig_layer auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{job_id} autoRig_layer auto_rig Request failed with status code"
            )
            return False, [None, None, []]
        # auto_rig_layer(mesh_output_path)

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{job_id} {os.path.join(mesh_output_path, 'mesh/mesh.glb')} not exist"
            )
            return False, [None, None, []]

        end_time = time.time()
        self.logger.info(f"{job_id} auto_rig cost time: {end_time - start_time} s")

        start_time = time.time()
        self.logger.info(f"{job_id} mesh_output_path:{mesh_output_path}")
   
        glb_path = os.path.join(mesh_output_path, "mesh/mesh.glb")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        try:
            animation_gif(glb_path, gif_path, self.logger)
        except:
            self.logger.error(f"{job_id} animation error")
            return False, [None, None, []]

        end_time = time.time()
        self.logger.info(f"{job_id} animation cost time: {end_time - start_time} s")

        final_out = [
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            os.path.join(mesh_output_path, "mesh/mesh_animation.glb"),
            out_gif,
        ]
        json_str = ujson.dumps(final_out)
        # self.redis_conn.set(f'result_long_retrive_npc_all_auto:{long_prompt}', json_str)
        # self.redis_conn.publish('result_long_retrive_npc_all_auto', json.dumps({'method': 'long_retrive_npc_all_auto', 'args': [long_prompt], 'result': final_out}))

        return True, json_str

    # retrive_npc_all_auto
    def exposed_retrive_npc_all_auto(
        self,
        path_list,
        key_list,
        job_id,
        texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color="",
        action="running"
    ):
        """
        npc text retrieve
        input:txt example('The woman walked down the street with a confident stride, her black leather jacket hugging her curves in all the right places. She wore a simple white t-shirt underneath, tucked into a pair of high-waisted blue jeans that accentuated her long legs. Her black ankle boots clicked against the pavement as she made her way towards the cafe, and her oversized sunglasses shielded her eyes from the bright sun. A silver necklace with a small pendant hung around her neck, adding a touch of elegance to her otherwise casual outfit. She exuded a sense of effortless style and cool confidence that turned heads as she passed by.')
        output:glb path,glb path,glb folder path
        """

        if (key_list[1] == "" or key_list[2] == "") and key_list[4] == "":
            self.logger.error(f"{job_id} retrive_npc_all_auto:path_list not correct")
            return False, [None, None, []]

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
        self.logger.info(f"{job_id} paths_temp_in:{paths_temp_in}")

        if job_id != None:
            mesh_output_path = model_save_folder+ str(job_id)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+ str(unique_id)

        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)

        start_time = time.time()
        wrap_cloth_result_flag = self.wrap_cloth(mesh_output_path, paths_temp_in,job_id)
        if wrap_cloth_result_flag==None:
            self.logger.error(f"{job_id} {mesh_output_path} wrap_cloth error")
            return False, [None, None, []]

        end_time = time.time()
        self.logger.info(f"{job_id} wrap_cloth cost time: {end_time - start_time} s")

        start_time = time.time()
        success_flag, result_meshs = self.combine_interface.blocking_call_combine(str(uuid.uuid4()),
                                                            mesh_output_path,
                                                            output_mesh_filename=os.path.join(mesh_output_path, "mesh/mesh.glb"))
        if success_flag:
            self.logger.info(f"{job_id} app_autoRig_layer combine Response")
        else:
            self.logger.erroe(
                f"{job_id} app_autoRig_layer combine Request failed with status code"
            )
            return False, [None, None, []]

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{job_id} app_autoRig_layer combine Request failed with status code"
            )
            return False, [None, None, []]

        end_time = time.time()
        self.logger.info(f"{job_id} combine cost time: {end_time - start_time} s")

        start_time = time.time()
        success, out_gif = self.blender_cvt_interface.blocking_call_render_gif(init_job_id(), os.path.join(mesh_output_path, "mesh/mesh.glb"), os.path.join(mesh_output_path, "mesh/mesh.gif"))
        if success:
            self.logger.info(f"{job_id} blocking_call_render_gif success")
        else:
            self.logger.error(
                f"{job_id} blocking_call_render_gif failed"
            )
            return False, [None, None, []]
        
        end_time = time.time()
        print("blocking_call_render_gif cost time: {:.2f} s".format(end_time - start_time))

        start_time = time.time()
        self.logger.info(f"{job_id} mesh_output_path:{mesh_output_path}")

        glb_path = os.path.join(mesh_output_path, "mesh/mesh.glb")
        animation_path = os.path.join(mesh_output_path, "mesh/mesh_animation.glb")
        try:
            animation_text(glb_path, action, self.logger)
        except:
            self.logger.error(f"{job_id} animation error")
            return False, [None, None, []]
        end_time = time.time()
        self.logger.info(f"{job_id} animation cost time: {end_time - start_time} s")

        final_out = [
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            animation_path,
            [mesh_output_path],
            out_gif
        ]
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_retrive_npc_all_auto:{path_list}:{key_list}:{texture_replace}:{gender}:{shape}:{hair_color}', json_str)
        # self.redis_conn.publish('result_retrive_npc_all_auto', json.dumps({'method': 'retrive_npc_all_auto', 'args': [path_list,key_list,texture_replace,gender,shape,hair_color], 'result': final_out}))

        return True, json_str

    # retrive_npc_dislike
    def exposed_retrive_npc_dislike(self, mesh_output_path,job_id):
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
            "/aigc_cfs_gdp/xiaqiangdai/logs/dislike.txt",
            "a+",
        )
        f.write(templist[0])
        f.write("\n")
        f.close()

        # self.redis_conn.set(f'result_retrive_npc_dislike:{mesh_output_path}', '')
        # self.redis_conn.publish('result_retrive_npc_dislike', json.dumps({'method': 'retrive_npc_dislike', 'args': [mesh_output_path], 'result': ''}))
        return True, None

    # retrive_npc_animation_dislike
    def exposed_retrive_npc_animation_dislike(self, mesh_output_path,job_id):
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
            "/aigc_cfs_gdp/xiaqiangdai/logs/animation_dislike.txt",
            "a+",
        )
        f.write(templist[0])
        f.write("\n")
        f.close()
        # self.redis_conn.set(f'result_retrive_npc_animation_dislike:{mesh_output_path}', '')
        # self.redis_conn.publish('result_retrive_npc_animation_dislike', json.dumps({'method': 'retrive_npc_animation_dislike', 'args': [mesh_output_path], 'result': ''}))
        return True, None

    # retrive_npc_auto_binding_animation
    def exposed_retrive_npc_auto_binding_animation(self, mesh_output_path,job_id):
        """
        retrive_npc_auto_binding_animation
        input:mesh_output_path
        output:glb path,gif path,glb folder path
        """

        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}

        res = requests.post(
            "http://url:8080/app_autoRig_layer/auto_rig",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{job_id} autoRig_layer auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{job_id} autoRig_layer auto_rig Request failed with status code"
            )
            return False, [None, None, []]

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{job_id} {os.path.join(mesh_output_path, 'mesh/mesh.glb')} not exist"
            )
            return False, [None, None, []]

        self.logger.info(f"{job_id} mesh_output_path:{mesh_output_path}")

        glb_path = os.path.join(mesh_output_path, "mesh/mesh.glb")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        try:
            animation(glb_path, gif_path, self.logger)
        except:
            self.logger.error(f"{job_id} animation error")
            return False, [None, None, []]

        final_out = [
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            gif_path,
            [mesh_output_path],
        ]
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_retrive_npc_auto_binding_animation:{mesh_output_path}', json_str)
        # self.redis_conn.publish('result_retrive_npc_auto_binding_animation', json.dumps({'method': 'retrive_npc_auto_binding_animation', 'args': [mesh_output_path], 'result': final_out}))

        return True, json_str

    # retrive_npc_combine
    def exposed_retrive_npc_combine(
        self,
        path_list,
        key_list,
        job_id,
        texture_replace=[False, False, False, False, False, False],
        gender="male",
        shape="fat",
        hair_color="gold",
        scope = "webui",
        is_combined_first=True
    ):
        """
        retrive_npc_manual_binding
        input:txt example(['/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Female hair/Female hair/F_HAIR_346/F_HAIR_346_fbx2020.glb',None, '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Bottoms/Bottoms/Bottoms01/BTM_93/BTM_93_fbx2020.glb', '/mnt/business_1/Data/DesignCenter/clock_fix_sample/20231215/fix_top_bottom/component/shoe/SK_Shoe_Sneaker03_F/SK_Shoe_Sneaker03_F.glb','/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/modify/Dresses/F_A/DR_673_F_A/DR_673_fbx2020.glb',  '/aigc_cfs/Asset/designcenter/clothes/mesh/designcenter_part2/clothes/Glove Socks/Glove Socks/Socks/SOX_129/SOX_129_fbx2020.glb'],[F_HAIR_346_Asset,None,DSBA_BTM_4_Bottoms03,SH_241_SHOES01,DR_673_F_A_Dresses,SOX_204_SOX])
        if not be selected,that shoud be None
        output:glb folder path,glb path

        """
        if (key_list[1] == "" or key_list[2] == "") and key_list[4] == "":
            self.logger.error(f"{job_id} retrive_npc_combine:path_list not correct")
            return False, ujson.dumps([None, None])

        paths_temp = {}
        attr_keys = ["hair", "top", "trousers", "shoe", "outfit", "others"]
        self.logger.info(
            f"{job_id} path_list:{path_list}  key_list:{key_list}  texture_replace:{texture_replace}"
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
            'is_combined_first':is_combined_first
        }
        self.logger.info(f"{job_id} paths_temp_in:{paths_temp_in}")

        if job_id != None:
            mesh_output_path = model_save_folder+ str(job_id)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+ str(unique_id)
        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)
            self.logger.info(f"{job_id} {mesh_output_path} not exists,make")
        else:
            self.logger.info(f"{job_id} {mesh_output_path}  exists")

        start_time = time.time()
        wrap_cloth_result_flag = self.wrap_cloth(mesh_output_path, paths_temp_in,job_id)
        if wrap_cloth_result_flag==None:
            self.logger.error(f"{job_id} {mesh_output_path} wrap_cloth error")
            return False, ujson.dumps([None, None])
        
        end_time = time.time()
        self.logger.info(f"{job_id} wrap_cloth cost time: {end_time - start_time} s")

        self.logger.info(f"{job_id} mesh_output_path:{mesh_output_path}")

        start_time = time.time()
        success_flag, result_meshs = self.combine_interface.blocking_call_combine(str(uuid.uuid4()),
                                                            mesh_output_path,
                                                            output_mesh_filename=os.path.join(mesh_output_path, "mesh/mesh.glb"))
        if success_flag:
            self.logger.info(f"{job_id} app_autoRig_layer combine Response")
        else:
            self.logger.error(
                f"{job_id} app_autoRig_layer combine Request failed with status code {success_flag}"
            )
            return False, ujson.dumps([None, None])

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{job_id} {os.path.join(mesh_output_path, 'mesh/mesh.glb')} not exist"
            )
            return False, ujson.dumps([None, None])

        end_time = time.time()
        self.logger.info("combine cost time: {:.2f} s".format(end_time - start_time))

        if scope == "webui":
            start_time = time.time()
            success, out_gif = self.blender_cvt_interface.blocking_call_render_gif(init_job_id(), os.path.join(mesh_output_path, "mesh/mesh.glb"), os.path.join(mesh_output_path, "mesh/mesh.gif"))
            if success:
                self.logger.info(f"{job_id} blocking_call_render_gif success")
            else:
                self.logger.error(
                    f"{job_id} blocking_call_render_gif failed"
                )
                return False, [None, None, []]
            
            end_time = time.time()
            self.logger.info("blocking_call_render_gif cost time: {:.2f} s".format(end_time - start_time))
        else:
            out_gif = None

        json_str = ujson.dumps(
            [mesh_output_path, os.path.join(mesh_output_path, "mesh/mesh.glb"),out_gif]
        )
        
        return True, json_str

    # retrive_npc_autoRig
    def exposed_retrive_npc_autoRig(self, mesh_output_path,job_id):
        """
        retrive_npc_autoRig
        input:mesh folder path
        output:mesh folder path,glb path
        """
        self.logger.info(f"{job_id} mesh_output_path:{mesh_output_path}")
        input = {"folder": mesh_output_path}
        json_data = json.dumps(input)
        headers = {"Content-Type": "application/json"}

        res = requests.post(
            "http://url:8080/app_autoRig_layer/auto_rig",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{job_id} autoRig_layer auto_rig Response:{res}")
        else:
            self.logger.error(
                f"{job_id}autoRig_layer auto_rig Request failed with status code"
            )
            return False, [None, None, []]

        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{job_id} {os.path.join(mesh_output_path, 'mesh/mesh.glb')} not exist"
            )
            return False, ujson.dumps([None, None, []])

        final_out = [mesh_output_path, os.path.join(mesh_output_path, "mesh/mesh.glb")]
        json_str = ujson.dumps(final_out)
        # self.redis_conn.set(f'result_retrive_npc_autoRig:{mesh_output_path}', json_str)
        # self.redis_conn.publish('result_retrive_npc_autoRig', json.dumps({'method': 'retrive_npc_autoRig', 'args': [mesh_output_path], 'result': final_out}))

        return True, json_str

        return True, json_str

    # retrive_npc_manual_binding
    def exposed_retrive_npc_manual_binding(self, mesh_output_path,job_id):
        """
        retrive_npc_manual_binding
        input:mesh folder path
        output:mesh folder path

        todo
        """

        # self.redis_conn.set(f'result_retrive_npc_manual_binding:{mesh_output_path}', '')
        # self.redis_conn.publish('result_retrive_npc_manual_binding', json.dumps({'method': 'retrive_npc_manual_binding', 'args': [mesh_output_path], 'result': ''}))
        return True, mesh_output_path

    # retrive_npc_animation
    def exposed_retrive_npc_animation(self, model_path,job_id):
        """
        retrive_npc_animation
        input:model_path
        output:glb path,gif path,glb folder path

        """
        start_time = time.time()
        gif_path = model_path.replace(".fbx","_animation.gif").replace(".glb","_animation.gif")
        try:
            animation(model_path, gif_path)
        except:
            self.logger.error(f"retrive_npc_animation error {model_path}")
            return False, [None, None, []]
        end_time = time.time()
        self.logger.info(f"retrive_npc_animation {job_id} cost time: {end_time - start_time} s")
        final_out = [
            model_path.replace(".fbx","_animation.fbx").replace(".glb","_animation.glb"),
            gif_path,
            [mesh_output_path],
        ]
        json_str = ujson.dumps(final_out)
        # self.redis_conn.set(f'result_retrive_npc_animation:{mesh_output_path}', json_str)
        # self.redis_conn.publish('result_retrive_npc_animation', json.dumps({'method': 'retrive_npc_animation', 'args': [mesh_output_path], 'result': final_out}))

        return True, json_str

    # retrive_npc_text_animation
    def exposed_retrive_npc_text_animation(self, model_path, text_prompt,job_id):
        """
        retrive_npc_animation
        input:glb  path,text_prompt
        output:glb path,gif path,glb folder path
        """
        start_time = time.time()
        if not os.path.exists(model_path):
            self.logger.error(f"retrive_npc_text_animation error {model_path} not exist")
            return False, [None, None, []]
        try:
            animation_text(model_path, text_prompt)
        except:
            self.logger.error(f"retrive_npc_text_animation error {model_path} {text_prompt}")
            return False, [None, None, []]
        end_time = time.time()
        self.logger.info(f"retrive_npc_text_animation {job_id} cost time: {end_time - start_time} s")

        final_out = model_path.replace(".fbx","_animation.fbx").replace(".glb","_animation.glb")
        json_str = ujson.dumps(final_out)


        return True, json_str
    
    # retrive_npc_animation_text_retrieve
    def exposed_npc_animation_text_retrieve(self, text_prompt,job_id):
        """
        retrive_npc_animation_text_retrieve
        input:text_prompt
        output:gif path
        """
        start_time = time.time()
        if text_prompt=='':
            self.logger.error(f"retrive_npc_animation_text_retrieve text null")
            return False, []
        try:
            ret = animation_text_retrieve(text_prompt)
        except:
            self.logger.error(f"retrive_npc_animation_text_retrieve error {text_prompt}")
            return False, []
        end_time = time.time()
        self.logger.info(f"retrive_npc_animation_text_retrieve {job_id} cost time: {end_time - start_time} s")

        json_str = ujson.dumps(ret)


        return True, json_str
    
    # animation_gif_generate
    def exposed_animation_gif_generate(self, model_path, gif_path,job_id):
        """
        animation_gif
        input:glb  path,gif_path
        output:glb path
        """
        start_time = time.time()
        if not os.path.exists(model_path):
            self.logger.error(f"animation_gif error {model_path} not exist")
            return False, None
        self.logger.info(f"animation_gif_generate {job_id} {model_path} {gif_path}")
        try:
            output_filepath = animation_text_retrieve_retarget(gif_path,model_path)
        except:
            self.logger.error(f"animation_gif_generate error {model_path} {gif_path}")
            return False, None
        end_time = time.time()
        self.logger.info(f"animation_gif_generate {job_id} {output_filepath} cost time: {end_time - start_time} s")

        final_out = output_filepath
        json_str = ujson.dumps(final_out)


        return True, json_str

    # auto_rig_manual_render
    def exposed_auto_rig_manual_render(self, mesh_out_path,job_id):
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
            "http://url:8080/app_autoRig/manual_process_render",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            print("app_autoRig auto_rig Response:", res)
        else:
            print("app_autoRig auto_rig failed with status code")
            return False, None

        print(res)
        bone_pic = os.path.join(input_path, "show_image.png")
        jointsjson_path = os.path.join(input_path, "joints.json")

        final_out = [bone_pic, jointsjson_path]
        json_str = ujson.dumps(final_out)
        # self.redis_conn.set(f'result_auto_rig_manual_render:{mesh_out_path}', json_str)
        # self.redis_conn.publish('result_auto_rig_manual_render', json.dumps({'method': 'auto_rig_manual_render', 'args': [mesh_out_path], 'result': final_out}))

        return True, json_str

    # auto_rig_manual_calculate
    def exposed_auto_rig_manual_calculate(self, joints_data, mesh_out_path,job_id):
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
            "http://url:8080/app_autoRig/manual_process_calculate",
            data=json_data,
            headers=headers,
        )
        if res.status_code == 200:
            self.logger.info(f"{job_id} app_autoRig auto_rig Response:{res}")
        else:
            self.logger.info(
                f"{job_id} app_autoRig auto_rig failed with status code"
            )
            return False, None
        model_path = os.path.join(input_path, "mesh.glb")
        resp = {"model_path": model_path}

        final_out = resp
        json_str = ujson.dumps(final_out)
        # self.redis_conn.set(f'result_auto_rig_manual_calculate:{joints_data}:{mesh_out_path}', json_str)
        # self.redis_conn.publish('result_auto_rig_manual_calculate', json.dumps({'method': 'auto_rig_manual_calculate', 'args': [joints_data, mesh_out_path], 'result': final_out}))

        return True, json_str

    # init_chat
    def exposed_init_chat(self):
        self.logger.info(f"{job_id} exposed_init_chat begin")
        from chat_service import ChatService

        # chatservice = ChatService()
        # ret = chatservice.request_chat(0.2)
        # return chatservice, ret["data"]["response"]
        try:
            chatservice = ChatService()
            ret = chatservice.request_chat(0.2)

        except:
            self.logger.error(f"{job_id} exposed_init_chat fail")

            return False, None

        final_out = [chatservice, ret["data"]["response"]]
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_init_chat', json_str)
        # self.redis_conn.publish('result_init_chat', json.dumps({'method': 'init_chat', 'args': [], 'result': final_out}))

        return True, json_str

    # generateRole_with_chat
    def exposed_generateRole_with_chat(self, chatservice, text, img):

        self.logger.info(f"{job_id} exposed_generateRole_with_chat begin")

        chatservice.update_chat_history(text, img)
        try:
            ret = chatservice.request_summary()
        except:  # 总结失败
            self.logger.error(
                f"{job_id} exposed_generateRole_with_chat summary fail"
            )
            return False, None

            #  怎么处理？

        chatservice.update_history(text, img)

        # 请求下一轮对话成功
        try:
            ret = chatservice.request_chat(0.7)

        except:
            print("exposed_generateRole_with_chat chat fail")
            return False, None

        final_out = [ret["data"]["response"]]
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_generateRole_with_chat:{text}', json_str)
        # self.redis_conn.publish('result_generateRole_with_chat', json.dumps({'method': 'generateRole_with_chat', 'args': [text], 'result': final_out}))
        return True, json_str

    # wework_ibot_retrive_npc
    def exposed_wework_ibot_retrive_npc(self, long_prompt):
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
            f"{job_id} extract_entity_all cost time: {end_time - start_time} s"
        )

        start_time = time.time()
        strs = entity[:6]

        self.logger.info(f"{job_id} {long_prompt} {description} {entity}")
        self.logger.info(f"{job_id} befour:{strs} len:{len(strs)}")
        if len(strs) < 6 and len(strs) > 0:
            for i in range(6 - len(strs)):
                strs.append(strs[len(strs) - 1])
        elif len(strs) > 6:
            strs = strs[:6]

        strs = prompt_preprocess(strs)

        self.logger.info(f"{job_id} after:{strs} len:{len(strs)}")

        suit_enale = False
        if strs[4] != "":
            suit_enale = True
        if strs[4] == "":
            suit_enale = False
        gender = entity[-1]
        hair_color = entity[-2]
        body_shape = entity[-3]

        if gender not in genders:
            self.logger.info(f"{job_id} wework_ibot_retrive_npc gender error")
            return False, ujson.dumps([None])

        img_paths_out = []
        glb_paths_out = []
        keys_out = []
        final_out = []
        for key_id, s in enumerate(strs):
            keys = retrive_single_txt(s, gender + "_" + part_keys[key_id])

            None_num = keys.count(None)
            self.logger.info(f"{job_id} key:{s} None_num:{None_num}")

            # if None_num==3 and (''!=s and ' '!=s  and None!=s and 'None'!=s):
            if None_num == 3 and key_id!=0 and key_id!=3:
                keys = retrive_single_txt(
                    description[:70], gender + "_" + part_keys[key_id]
                )

            self.logger.info(f"{job_id} keys:{keys}")
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
        shape = "fat"
        paths_temp_in = {
            "path": paths_temp,
            "body_attr": [gender, body_shape],
            "hair_color": hair_color,
        }
        self.logger.info(f"{job_id} paths_temp_in:{paths_temp_in}")

        if job_id != None:
            mesh_output_path = model_save_folder+ str(job_id)
        else:
            timestamp = int(time.time())
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = model_save_folder+ str(unique_id)

        if not os.path.exists(mesh_output_path):
            os.makedirs(mesh_output_path)

        start_time = time.time()
        wrap_cloth_result_flag = self.wrap_cloth(mesh_output_path, paths_temp_in,job_id)
        if wrap_cloth_result_flag==None:
            self.logger.error(f"{job_id} {mesh_output_path} wrap_cloth error")
            return False, [None, None, []]

        end_time = time.time()
        self.logger.info(f"{job_id} wrap_cloth cost time: {end_time - start_time} s")

        start_time = time.time()
        success_flag, result_meshs = self.combine_interface.blocking_call_combine(str(uuid.uuid4()),
                                                            mesh_output_path,
                                                            output_mesh_filename=os.path.join(mesh_output_path, "mesh/mesh.glb"))
        if success_flag:
            self.logger.info(f"{job_id} app_autoRig_layer combine Response")
        else:
            self.logger.error(
                f"{job_id} app_autoRig_layer combine Request failed with status code"
            )
            return False, [None, None, []]
        if not os.path.exists(os.path.join(mesh_output_path, "mesh/mesh.glb")):
            self.logger.error(
                f"{job_id} {os.path.join(mesh_output_path, 'mesh/mesh.glb')} not exist"
            )
            return False, [None, None, []]

        end_time = time.time()
        self.logger.info(f"{job_id} combine cost time: {end_time - start_time} s")

        final_out = [
            os.path.join(mesh_output_path, "mesh/mesh.glb"),
            path_list,
            key_list,
            gender,
            shape,
        ]
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_wework_ibot_retrive_npc:{long_prompt}', json_str)
        # self.redis_conn.publish('result_wework_ibot_retrive_npc', json.dumps({'method': 'wework_ibot_retrive_npc', 'args': [long_prompt], 'result': final_out}))

        return True, json_str

    # wework_ibot_animation
    def exposed_wework_ibot_animation(self, mesh_output_path):
        start_time = time.time()
        print(mesh_output_path)

        glb_path = os.path.join(mesh_output_path, "mesh/mesh.glb")
        gif_path = os.path.join(mesh_output_path, "mesh/mesh_animation.gif")
        try:
            animation_gif(glb_path, gif_path, self.logger)
        except:
            self.logger.error(f"{job_id} gif animation error")
            return False, [None, None, []]
        end_time = time.time()
        self.logger.info(f"{job_id} animation cost time: {end_time - start_time} s")

        final_out = [
            os.path.join(mesh_output_path, "mesh/mesh_animation.glb"),
            gif_path,
        ]
        json_str = ujson.dumps(final_out)

        # self.redis_conn.set(f'result_wework_ibot_animation:{mesh_output_path}', json_str)
        # self.redis_conn.publish('result_wework_ibot_animation', json.dumps({'method': 'wework_ibot_animation', 'args': [mesh_output_path], 'result': final_out}))

        return True, json_str

    def exposed_render_gif_frontImage(self, model_path,job_id):
        """
        render_gif_frontImage
        input:model_path
        output:360 video and front image

        """
        self.logger.info(f"{job_id}  model_path:{model_path}")
        model_folder = os.path.dirname(model_path)
        render_foler = os.path.join(model_folder, "render")
        if not os.path.exists(render_foler):
            os.mkdir(render_foler)

        cmd1 = f"/root/blender-3.6.5-linux-x64/blender -b -P \
        /aigc_cfs_2/xiaqiangdai/project/render_gif_frontImage/render_color_depth_normal_helper.py -- \
        --mesh_path {model_path}  \
        --output_folder {render_foler} \
        --pose_json_path /mnt/aigc_cfs_cq/xiaqiangdai/project/render_gif_frontImage/20240407-S-120-cam_parameters.json  \
        --white_background  --no_solidify --save_gif  --engine eevee --render_height 256 --render_width 256 --only_render_png  --smooth "
        os.system(cmd1)
        os.system("rm " + render_foler + "/color/*png")
        cmd2 = f"/root/blender-3.6.5-linux-x64/blender -b -P \
        /aigc_cfs_2/xiaqiangdai/project/render_gif_frontImage/render_color_depth_normal_helper.py -- \
        --mesh_path {model_path}  \
        --output_folder {render_foler} \
        --pose_json_path /mnt/aigc_cfs_cq/xiaqiangdai/project/render_gif_frontImage/20240328-S-1-cam_parameters.json  \
        --white_background  --no_solidify  --engine eevee --render_height 256 --render_width 256 --only_render_png  --smooth "
        os.system(cmd2)
        final_out = [
            os.path.join(render_foler, "color/cam-0000.png"),
            os.path.join(render_foler, "color/render.webm"),
        ]
        json_str = ujson.dumps(final_out)

        return True, json_str

    def exposed_render_shader(self, model_path,job_id):
        """
        render_shader
        input:model_path
        output:shader video and front image
        """
        self.logger.info(f"{job_id}  render_shader model_path:{model_path}")
        model_folder = os.path.dirname(model_path)
        render_foler = os.path.join(model_folder, "render")
        start_time = time.time()
        RenderMesh(model_path)
        render2video(model_path)
        end_time = time.time()
        self.logger.info(f"render_shader {job_id} {model_path} cost time: {end_time - start_time} s")
        if not os.path.exists(
            os.path.join(render_foler, "CoverCollect.png")
        ) or not os.path.exists(os.path.join(render_foler, "render.webm")):
            self.logger.error(
                f"{job_id} render_shader model_path:{model_path} error"
            )
            return False, ujson.dumps([None, None])
        final_out = [
            os.path.join(render_foler, "CoverCollect.png"),
            os.path.join(render_foler, "render.webm"),
        ]
        json_str = ujson.dumps(final_out)

        return True, json_str

    def exposed_shape_text_retrieve(self, gender, shape_promt,job_id):
        """
        shape_text_retrieve
        input:gender, shape_promt
        output:gender,shape,use_shoes,use_hair
        """
        start_time = time.time()
        body_attr = shape_text_retrieve(gender, shape_promt)

        end_time = time.time()
        self.logger.info(f"shape_text_retrieve {job_id} {shape_promt} cost time: {end_time - start_time} s")

        use_shoes = base_body_map[gender][body_attr[1]]["use_shoes"]
        use_hair = base_body_map[gender][body_attr[1]]["use_hair"]
        final_out = {
            "gender": gender,
            "shape": body_attr[1],
            "use_shoes": use_shoes,
            "use_hair": use_hair,
        }
        json_str = ujson.dumps(final_out)

        return True, json_str

    def exposed_hair_is_change_ok(self, mesh_path):
        """
        hair_is_change_ok
        input:mesh_path
        output:True or False
        """
        try:
            ret = is_change_ok_online(mesh_path)
        except Exception as e:
            self.logger.error(f"{self.uuid} hair_is_change_ok error")
            return False,ujson.dumps(False)
        json_str = ujson.dumps(ret)
        return True, json_str

    def exposed_get_retrieve_json_path(self):
        """
        get_retrieve_json_path
        input:
        output:cfs and gdp json path
        """
        final_out = [json_path, gdp_json_path]
        json_str = ujson.dumps(final_out)
        return True, json_str


if __name__ == "__main__":
    RetrieveNPCService()
    app.run(host='0.0.0.0', port=8081)
    while True:
        time.sleep(60)
        
