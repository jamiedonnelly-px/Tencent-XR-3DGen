import sys
import json
import threading
from queue import Queue
import time
import ujson
import uuid
import redis
from datetime import datetime
import os
codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)
from distribute_logging import *


def serialize(obj):
    if isinstance(obj, uuid.UUID):
        return str(obj)
    else:
        return json.JSONEncoder().default(obj)

class retrieve_npc_service():
    def __init__(self,host="",password='',port=6379,db_id=0):
        self.redis_conn_pool = redis.ConnectionPool(host=host,password=password,port=port,db=db_id,max_connections=6)
        self.message_types = ['cos_upload','cos_upload_remote','txt_retrive_npc','long_txt_retrive_npc','long_retrive_npc_all_auto','retrive_npc_all_auto','retrive_npc_dislike','retrive_npc_animation_dislike','retrive_npc_auto_binding_animation','retrive_npc_combine','retrive_npc_autoRig','retrive_npc_animation','retrive_npc_text_animation',"npc_animation_text_retrieve","animation_gif_generate",'render_shader','shape_text_retrieve','hair_is_change_ok','get_retrieve_json_path']
        self.queue = []
        self.output_results=[]
        get_results_thread = threading.Thread(target=self.get_results)
        get_results_thread.daemon = True
        get_results_thread.start()

        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day

        self.log_file_path = (
            f"/aigc_cfs_gdp/xiaqiangdai/logs/retrieveNPC_redis/{year}_{month}_{day}.log"
        )
        self.logger = DistributedFileLogger("RetrieveNPC_redis_logger", self.log_file_path)

    def connect_redis(self):
        while True:
            try:
                r = redis.Redis(connection_pool=self.redis_conn_pool)
                r.ping()
                return r
            except redis.exceptions.ConnectionError as e:
                self.logger.error("连接失败，尝试重新连接...")
                time.sleep(5)
    
    def submit_task(self,message_type,task_id,input_param):
        if message_type not in self.message_types:
            self.logger.error(f"message_type error {message_type} {task_id}")
            return None
        task_id = str(task_id)
        self.logger.info(f"submit_task {message_type} {task_id}")
        message = {'channel':message_type,'data':input_param,'id':task_id}
        m_str = ujson.dumps(message,default=serialize)

        redis_conn = self.connect_redis()
        self.logger.info(f"npc_retrieve_queue length:{redis_conn.llen('npc_retrieve_queue')}")
        redis_conn.lpush("npc_retrieve_queue", m_str)
        redis_conn.hset('message_status', '_'.join([message_type,task_id]), "pending")
        self.logger.info(f"npc_retrieve_queue length:{redis_conn.llen('npc_retrieve_queue')}")
        input_message = {"message_type":message_type,"task_id":task_id}
        self.queue.append(input_message)
        return 0

    def get_task_status(self,message_type,task_id):
        if message_type not in self.message_types:
            self.logger.error(f"message_type error {message_type} {task_id}")
            return None
        task_id = str(task_id)
        redis_conn = self.connect_redis()
        status = redis_conn.hget('message_status', '_'.join([message_type,task_id]))
        if status!=None:
            return status.decode()[:]
        else:
            return ''
    
    
    def get_task_wait_num(self,message_type,task_id):
        if message_type not in self.message_types:
            self.logger.error(f"message_type error {message_type} {task_id}")
            return None

        redis_conn = self.connect_redis()
        message_list = redis_conn.lrange("npc_retrieve_queue", 0, -1)
        position = 0
        for i, message_str in enumerate(message_list):
            message = ujson.loads(message_str)
            if message['channel'] == message_type and message['id'] == task_id:
                self.logger.info(f"Position of message with channel '{message_type}' and task_id '{task_id}': {i}")
                position = i
                break
        return position
    
    def get_task_num(self):
        redis_conn = self.connect_redis()
        message_list = redis_conn.lrange("npc_retrieve_queue", 0, -1)
        return len(message_list)

    def get_single_result(self,message_type,task_id):
        if message_type not in self.message_types:
            self.logger.error(f"message_type error {message_type} {task_id}")
            return None
        channel = 'result_'+message_type+'_'+str(task_id)
        
        redis_conn = self.connect_redis()
        result = redis_conn.get(channel.encode('utf-8'))
        if result:
            return ujson.loads(result.decode())
        else:
            return None

    def get_results(self):
        while True: 
            try:
                items_to_remove = []
                for item in self.queue:
                    result = self.get_single_result(item['message_type'],item['task_id'])
                    if result!=None:
                        items_to_remove.append(item)
                        key = '_'.join([item['message_type'],item['task_id']])
                        self.output_results.append({key:result})
                        redis_conn = self.connect_redis()
                        redis_conn.hset('message_status', '_'.join([item['message_type'],item['task_id']]), "received")
                        for i in range(5):
                            status = redis_conn.hget('message_status', '_'.join([item['message_type'],item['task_id']])).decode()
                            if status!='received':
                                redis_conn.hset('message_status', '_'.join([item['message_type'],item['task_id']]), "received")
                                self.logger.info(f"set status received:{status}")
                            else:
                                break
                            time.sleep(0.1)
                            
                for item in items_to_remove:
                    self.queue.remove(item)

                if len(self.output_results)>1000:
                    del self.output_results[:100]

                time.sleep(1)
            except Exception as e:
                self.logger.error(f"retrieve_npc_backend get_results error {e}")
    
    def user_get_result(self,message_type,task_id):
        task_id = str(task_id)
        self.logger.info(f"user_get_result:{message_type} {task_id}")
        if message_type not in self.message_types:
            self.logger.error(f"message_type error {message_type} {task_id}")
            return None

        get_num = 0
        while True:
            status = self.get_task_status(message_type,task_id)
            # print(status,":",type(status),len(status),message_type,task_id)
            
            if status=="received":
                key = '_'.join([message_type,task_id])
                for i,item in enumerate(self.output_results):
                    if key ==list(item.keys())[0]:
                        result = self.output_results[i][key]['result']
                        self.logger.info(f"{message_type} {task_id} user_get_result sucess")
                        return result
            if status=="failed":
                return None
            get_num+=1
            time_limit = 600
            if get_num>time_limit:
                self.logger.error(f"user_get_result beyong limit time {message_type} {task_id}")
                return None
            time.sleep(1)


if __name__ == "__main__":

    service = retrieve_npc_service()
    print(service)
   
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    print(unique_id)
    unique_id_str = str(unique_id)
    
    message = {'channel':'long_txt_retrive_npc','data':['a girl with black hair',unique_id_str],'id':unique_id_str}

    # m_str = ujson.dumps(message)
   
    service.submit_task(message['channel'],message['id'],message['data'])

    while True:
        status = service.get_task_status(message['channel'],message['id'])
        print(status)
        wait_num = service.get_task_wait_num(message['channel'],message['id'])
        print(wait_num)
        time.sleep(1)
        if status=='received':
            result = service.user_get_result(message['channel'],message['id'])
            print(result)
            break
        

        