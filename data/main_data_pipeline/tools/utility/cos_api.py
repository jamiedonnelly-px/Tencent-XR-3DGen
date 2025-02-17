import os
import sys
from concurrent.futures import ThreadPoolExecutor

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos import CosServiceError


# pip install -U cos-python-sdk-v5 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


class CosClient():
    def __init__(self, secret_id: str, secret_key: str):
        region = 'ap-chongqing'
        domain = '3d-aigc-1258344700.cos-internal.ap-chongqing.tencentcos.cn'
        self.bucket = '3d-aigc-1258344700'
        token = None
        scheme = 'https'
        endpoint = 'cos-internal.%s.tencentcos.cn' % region
        service_domain = 'service.cos.tencentcos.cn'
        if len(secret_id) != 0 and len(secret_key) != 0:
            config = CosConfig(Secret_id=secret_id, Secret_key=secret_key, Token=token, Endpoint=endpoint,
                               Scheme=scheme, ServiceDomain=service_domain)
            # 创建COS客户端
            self.cosclient = CosS3Client(config)

    def check_exists(self, remote_path):
        try:
            response = self.cosclient.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except Exception as e:
            return False

    def upload(self, local_path, remote_path, get_url=True):
        # 上传本地文件到COS
        response = self.cosclient.upload_file(Bucket=self.bucket, LocalFilePath=local_path, Key=remote_path,
                                              PartSize=10, MAXThread=10, progress_callback=None)
        url = None
        if get_url:
            url = self.cosclient.get_presigned_url(Method='GET', Bucket=self.bucket, Key=remote_path,
                                                   Expired=24 * 60 * 60 * 360 * 20)
        return url

    def download(self, local_path, remote_path, get_url=True):
        response = self.cosclient.download_file(
            Bucket=self.bucket,
            Key=remote_path,
            DestFilePath=local_path)

        url = None
        if get_url:
            url = self.cosclient.get_presigned_url(Method='GET', Bucket=self.bucket, Key=remote_path,
                                                   Expired=24 * 60 * 60)
        return url

    def create_folder(self, folder_name: str):
        response = self.cosclient.put_object(Bucket=self.bucket, Key=folder_name, Body=b'')
        print(response)

    def exists(self, file_path: str):
        exists = False
        try:
            response = self.cosclient.head_object(Bucket=self.bucket, Key=file_path)
            exists = True
        except CosServiceError as e:
            if e.get_status_code() == 404:
                exists = False
            else:
                print("Error happened, reupload it.")
        return exists


def generate_upload_threads(data_folder: str,
                            new_upload_path: str,
                            cos_client: CosClient,
                            thread_pool: ThreadPoolExecutor):
    local_file_cos_file_map = {}
    for path, dir_list, file_list in os.walk(data_folder):
        for file_name in file_list:
            current_file_path = os.path.join(path, file_name)
            local_file_key = os.path.relpath(current_file_path, data_folder)
            if sys.platform.startswith('win'):
                clean_file_subpath = local_file_key.strip('\\')
                cos_file_subpath = clean_file_subpath.replace('\\', '/')
            else:
                cos_file_subpath = local_file_key.strip('/')

            cos_file_fullpath = new_upload_path + "/" + cos_file_subpath
            print("Upload file from %s (relative path is %s) to path %s" %
                  (current_file_path, local_file_key, cos_file_fullpath))
            thread_pool.submit(cos_client.upload, current_file_path, cos_file_fullpath)
            local_file_cos_file_map[current_file_path] = cos_file_fullpath
    return local_file_cos_file_map
