import pulsar
from _pulsar import ConsumerType, LoggerLevel
import uuid
import json
import logging
import time
import argparse
import os
import numpy as np
import threading
import torch
from PIL import Image
from easydict import EasyDict as edict

import redis

codedir = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')



if __name__ == "__main__":
    image_npy_path = '/aigc_cfs_gdp/neoshang/release/zero123plus/zero123plus_v4.4.1/log/7fe8287b-a7e7-4609-8203-4e4372b28777/mario.npy'
    image_np = np.load(image_npy_path)
    for i, img_arr in enumerate(image_np):
        img_arr = np.transpose(img_arr.clip(0, 255).astype(np.uint8), (1, 2, 0))
        img = Image.fromarray(img_arr)
        img.save(str(i)+".png")