# from gradio_client import Client
import sys
import rpyc
import json
import threading
from queue import Queue
import time
import ujson
import uuid
import redis
# from ipdb import set_trace as st

# 27603
def connect_redis():
    while True:
        try:
            r = redis.Redis(host='', password='',port=6379, db=0,health_check_interval=30)
            # r = redis.Redis(host='localhost', port=6379, db=0,health_check_interval=30)
            r.ping()
            return r
        except redis.exceptions.ConnectionError as e:
            print("连接失败，尝试重新连接...")
            time.sleep(5)

if __name__ == "__main__":
    
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 0,config=rpyc_config)
    print(connection)
    timestamp = int(time.time())
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
    print(unique_id)
    unique_id_str = str(unique_id)

 
    
    # message['cos_upload','texReplace','txt_retrive_npc','long_txt_retrive_npc','long_retrive_npc_all_auto','retrive_npc_all_auto','retrive_npc_dislike','retrive_npc_animation_dislike','retrive_npc_auto_binding_animation','retrive_npc_combine',
    # 'retrive_npc_autoRig','retrive_npc_manual_binding','retrive_npc_animation','retrive_npc_text_animation','auto_rig_manual_render','auto_rig_manual_calculate',
    # 'init_chat','generateRole_with_chat','wework_ibot_retrive_npc','wework_ibot_animation','render_gif_frontImage','render_shader','shape_text_retrieve','hair_is_change_ok','get_retrieve_json_path']
    
    message = {'channel':'long_txt_retrive_npc','data':['a girl'],'id':unique_id_str}
    m_str = ujson.dumps(message)
    
    connection.root.add_to_queue(m_str)
    connection.close()
    
    redis_conn = connect_redis()
    print(redis_conn)
    channel = 'result_'+message['channel']+'_'+str(message['id'])
    print(channel)
    pubsub = redis_conn.pubsub()

    pubsub.subscribe(channel)
    
    
    while True:
        message = pubsub.get_message()
        if message:
            print("接收到消息：", message)
        