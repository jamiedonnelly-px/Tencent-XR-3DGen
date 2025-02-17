import json
import sys
import os
from qcloud_cos import CosConfig, CosServiceError
from qcloud_cos import CosS3Client
from qcloud_cos.cos_threadpool import SimpleThreadPool

# self.character_remote = "/pandorax/character" #人物生成总路径
# self.object_remote = "/pandorax/object" #物体生成总路径
# self.retrieve_NPC_remote = "/pandorax/retrieve_NPC"  # 分层检索的总路径


class CosClient():
    

    def percentage(self,consumed_bytes, total_bytes):
        """进度条回调函数，计算当前上传的百分比

        :param consumed_bytes: 已经上传/下载的数据量
        :param total_bytes: 总数据量
        """
        if total_bytes:
            rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
            print('\r{0}% '.format(rate))
            sys.stdout.flush()
            
    def __init__(self):
        # 获取当前脚本的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取上一级目录的路径
        parent_dir = os.path.dirname(current_dir)
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
        secret_id = str(os.environ['COS_ACCESS_ID'])
        secret_key = str(os.environ['COS_SECRET_KEY'])    
        print(f'通过环境变量获取:{secret_id}===={secret_key}')
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
            response = self.cosclient.head_object(
                Bucket=self.bucket,
                Key=remote_path
            )
            return True
        except Exception as e:
            return False

    def upload(self, local_path, remote_path, get_url=True):
        # 上传本地文件到COS
        response = self.cosclient.upload_file(
            Bucket=self.bucket,
            LocalFilePath=local_path,
            Key=remote_path,
            PartSize=10,
            MAXThread=10,
            progress_callback=None
        )
        url = None
        if get_url:
            url = self.cosclient.get_presigned_url(
                Method='GET',
                Bucket=self.bucket,
                Key=remote_path,
                Expired=24 * 60 * 60 * 360 * 20  # 一天后过期，过期时间请根据自身场景定义
            )
        return url

    def custom_presigned_url(self, remote_path):
        url = self.cosclient.get_presigned_url(
            Method='GET',
            Bucket=self.bucket,
            Key=remote_path,
            Expired=24 * 60 * 60 * 360 * 20  # 一天后过期，过期时间请根据自身场景定义
        )
        return url

    """
    批量上传
    """
    def uploadMultiple(self, local_path_array, remote_path_array):
        # 创建上传的线程池
        urls = []
        try:
            upload_pool = SimpleThreadPool()
            for (index,local_path)  in enumerate(local_path_array):
                remote_path=remote_path_array[index]
                print(local_path,remote_path)
                if os.path.exists(local_path):
                    exists = False
                    try:
                        response = self.cosclient.head_object(Bucket=self.bucket, Key=remote_path)
                        exists = True
                    except CosServiceError as e:
                        if e.get_status_code() == 404:
                            exists = False
                        else:
                            print("Error happened, reupload it.")
                    if not exists:
                        print("File %s 不存在cos中, 上传它", local_path)
                        upload_pool.add_task(self.cosclient.upload_file, self.bucket,remote_path,local_path)
                    urls.append(self.custom_presigned_url(remote_path))
                else:
                    print(f"上传本地文件不存在跳过:{local_path}")
            upload_pool.wait_completion()
            result = upload_pool.get_result()
            print(result)
            if not result['success_all']:
                print("Not all files upload successed. you should retry")
            print(f"urls:{urls}")
        except Exception as ex:
             print("批量上传失败:{}".format(ex))   
        return urls
    
    



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

# 通过当前app上面取
# cos_client = CosClient()
    


    
    
    
    
    
   