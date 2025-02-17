import sys
import uuid
import time
import json
import requests
import os

import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer, ThreadPoolServer

rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
rpyc_config["sync_request_timeout"] = None
rpyc_config["allow_public_attrs"] = True
import argparse
import threading
import zlib
import pickle
import ujson
import logging
from datetime import datetime
import numpy as np
import sys
sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/npc_layer_retrieve/CLIP_img_gen")
from  single_retrieve_normic_test  import webui_retrieve


parser = argparse.ArgumentParser(description="启动所有RPC")
parser.add_argument("-H", "--hostname", default="0.0.0.0", help="外部设置ip")


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def deserialize(obj):
    if isinstance(obj, list):
        try:
            return np.array(obj)
        except ValueError:
            return [deserialize(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: deserialize(v) for k, v in obj.items()}
    else:
        return obj

class single_retrieve_service(Service):

    def __init__(self):
        self.sleepTime = 10
        self.processNum = 0
        self.uuid = None
        

    def on_connect(self, conn):
        # 连接建立时的回调函数
        print("rpyc 开始连接...")
        pass
            

    def on_disconnect(self, conn):
        # 连接断开时的回调函数
        print("rpyc 断开连接...")

   

    def exposed_retrive_text(self,input_json):
        input_data = ujson.loads(input_json)
        input_data = deserialize(input_data)
        text_prompt = input_data['text_prompt']
        gender = input_data['gender']
        part = input_data['part']
        output = webui_retrieve(text_prompt,gender,part)
        output = convert_to_serializable(output)
        return ujson.dumps(output)

        
        
if __name__ == "__main__":
    # while True:
    args = parser.parse_args()
    print("启动所有RPC:{}".format(args))

    server1 = ThreadPoolServer(single_retrieve_service, hostname=args.hostname, port=80,nbThreads=2)
    # server1 = ThreadPoolServer(
    #     RPCService, hostname=args.hostname1, port=8083, protocol_config=rpyc_config
    # )
    server1.start()
