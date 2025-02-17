from gradio_client import Client
import sys
import rpyc
import json
import threading
from queue import Queue
import time
import ujson
import ast
import os
import uuid
import argparse
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
# from ipdb import set_trace as st


parent_dir = "/aigc_cfs_2/jiawei/server_bakend"

def percentage(consumed_bytes, total_bytes):
    """进度条回调函数，计算当前上传的百分比

    :param consumed_bytes: 已经上传/下载的数据量
    :param total_bytes: 总数据量
    """
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate))
        sys.stdout.flush()

class CosClient():
    def __init__(self,secret_id,secret_key):
        self.assets_dir_remote = "/pandorax/assets"
        self.user_assets_dir_remote = "/pandorax/user_assets"
        self.user_images_dir_remote = "/pandorax/user_upload_images" #上传图片路径
        self.user_avatar_dir_remote = "/pandorax/user_avatars"  #头像路径
        self.generate_character_dir_remote = "/pandorax/character"
        self.upload_models_dir_remote = "/pandorax/upload_models" #搜索模型
        self.object_models_dir_remote = "/pandorax/object" #生成物体+搜索模型
        self.retrieve_NPC_dir_remote = "/pandorax/retrieve_NPC" #分层搜索模型
        self.retrieve_NPC_dir_imgs_remote = "/pandorax/retrieve_NPC/imgs"  # 分层搜索图片
        staticfolder=os.path.join(parent_dir,"static")
        self.user_images_local_dir = os.path.join(staticfolder, 'imgs')
        # os.makedirs(self.user_images_local_dir, exist_ok=True)

        region = 'ap-nanjing'
        domain = '3d-aigc-fast-1258344700.cos-internal.ap-nanjing.tencentcos.cn.'
        self.bucket = '3d-aigc-fast-1258344700'  # 存储桶名称
        token = None  # 使用临时密钥时，需要传入Token，默认为空
        # scheme = 'http'  #
        scheme = 'https'  # 指定使用 http/https 协议来访问 COS，默认为 https，可不填
        # 使用内网域名接入if
        endpoint = 'cos-internal.%s.tencentcos.cn' % region  # 桶域名(访问具体桶时使用)
        # endpoint = '3d-aigc-fast-1258344700.cos-internal.accelerate.tencentcos.cn'  # 桶域名(访问具体桶时使用)
        service_domain = 'service.cos.tencentcos.cn'  # 服务域名(列举所有桶时使用)(此域名暂时不支持内网域名，这里仍然使用的外网域名)
        if len(secret_id) != 0 and len(secret_key) != 0:
            config = CosConfig(Secret_id=secret_id, Secret_key=secret_key, Token=token, Endpoint=endpoint,
                               Scheme=scheme, ServiceDomain=service_domain)
            # 创建COS客户端
            self.cosclient = CosS3Client(config)

    def check_exists(self, remote_path):
        try:
            response = self.cos_client.head_object(
                Bucket=self.bucket,
                Key=remote_path
            )
            return True
        except Exception as e:
            return False

    def upload(self, local_path, remote_path, get_url=True):
        # 上传本地文件到COS
        print("进入上传接口")
        response = self.cosclient.upload_file(
            Bucket=self.bucket,
            LocalFilePath=local_path,
            Key=remote_path,
            PartSize=10,
            MAXThread=10,
            progress_callback=percentage
        )
        print(f"完成上传接口:{response}")
        url = None
        if get_url:
            print("获取链接")
            url = self.cosclient.get_presigned_url(
                Method='GET',
                Bucket=self.bucket,
                Key=remote_path,
                Expired=24 * 60 * 60 * 360 * 20  # 一天后过期，过期时间请根据自身场景定义
            )
            print(f"完成接口:{url}")
        return url
    
    

    def download(self, local_path, remote_path, get_url=True):
        response = self.cosclient.download_file(
            Bucket=self.bucket,
            Key=remote_path,
            DestFilePath=local_path)

        url = None
        if get_url:
            url = self.cosclient.get_presigned_url(
                Method='GET',
                Bucket=self.bucket,
                Key=remote_path,
                Expired=24 * 60 * 60*360  # 一天后过期，过期时间请根据自身场景定义
            )
        return url



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default='1')
    parser.add_argument("--key", type=str, default='2')
    args = parser.parse_args()
    print(args.id,args.key)
    # for i in range(10):
    guid = uuid.uuid4().hex
    local_pic_name = 'example5.jpg'
    new_pic_name = f'{guid}/{local_pic_name}'
    cos_client = CosClient(args.id,args.key)
    remote_file_path = os.path.join(cos_client.retrieve_NPC_dir_imgs_remote, new_pic_name)
    piclocalpath = '/aigc_cfs_2/jiawei/server_bakend/static/imgtemplate/obj1.png'
    piccosurl = cos_client.upload(piclocalpath, remote_file_path)
    print(f"piccosurl:{piccosurl}")
        
        
    # file_dir='/aigc_cfs_2/jiawei/server_bakend/static'
    # example_image_local_path_list = [file_dir + "/imgtemplate/person1.png",
    #                                  file_dir + "/imgtemplate/person2.png",
    #                                  file_dir + "/imgtemplate/person3.png",
    #                                  file_dir + "/imgtemplate/person4.png",
    #                                  file_dir + "/imgtemplate/obj1.png",
    #                                  file_dir + "/imgtemplate/obj2.png",
    #                                  file_dir + "/imgtemplate/obj3.png",
    #                                  file_dir + "/imgtemplate/obj4.png",
    #                                  ]
    # cos_client = CosClient(args.id,args.key)
    # for piclocalpath in example_image_local_path_list:
    #     print(piclocalpath)
    #     guid = uuid.uuid4().hex
    #     local_pic_name = os.path.basename(piclocalpath)
    #     new_pic_name = f'{guid}/{local_pic_name}'
    #     remote_file_path = os.path.join(cos_client.retrieve_NPC_dir_imgs_remote, new_pic_name)
    #     print(f"11:{piclocalpath}==22:{remote_file_path}")
    #     piccosurl = cos_client.upload(piclocalpath, remote_file_path)
    #     print(f"piccosurl:{piccosurl}")
   
