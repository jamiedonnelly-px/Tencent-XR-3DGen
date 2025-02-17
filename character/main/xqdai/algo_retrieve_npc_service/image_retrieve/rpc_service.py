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
from  single_retrieve_test  import single_retrieve


parser = argparse.ArgumentParser(description="启动所有RPC")
parser.add_argument("-H", "--hostname", default="0.0.0.0", help="外部设置ip")

retrieve_service = single_retrieve(model_type='bge')

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

   

    def exposed_retrive_image_all(self,input_json):
        input_data = ujson.loads(input_json)
        input_data = deserialize(input_data)
        input_image = np.array(input_data['image']).astype(np.uint8)
        print(input_image.shape)
        print(input_image.dtype)
        gender = input_data['gender']
        output = retrieve_service.retrive_image_all(input_image,gender)
        output = convert_to_serializable(output)
        return ujson.dumps(output)


    def exposed_retrieve_parts(self,input_json):
        input_data = ujson.loads(input_json)
        input_data = deserialize(input_data)
        input_image = np.array(input_data['image']).astype(np.uint8)
        print(input_image.shape)
        print(input_image.dtype)
        gender = input_data['gender']
        part_info = input_data['part_info']
        output = retrieve_service.retrieve_parts(input_image,part_info,gender)
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
