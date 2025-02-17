import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos import CosServiceError


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
            response = self.cos_client.head_object(
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
                Expired=24 * 60 * 60 * 360 * 20
            )
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
                Expired=24 * 60 * 60
            )
        return url

    def create_folder(self, folder_name: str):
        response = self.cosclient.put_object(Bucket=self.bucket,
                                             Key=folder_name,
                                             Body=b'')
        print(response)

    def exists(self, file_path: str):
        exists = False
        try:
            response = self.cosclient.head_object(
                Bucket=self.bucket, Key=file_path)
            exists = True
        except CosServiceError as e:
            if e.get_status_code() == 404:
                exists = False
            else:
                print("Error happened, reupload it.")
        return exists


def decorate_space(path_str: str):
    new_path_str = path_str.replace(" ", "\\ ")
    return new_path_str


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def read_list(in_list_txt):
    str_list = []
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        return str_list

    with open(in_list_txt, 'r', encoding='UTF-8') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w', encoding='UTF-8') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def generate_thumbnail_list(render_data_folder: str, thumbnail_folder: str):
    render_log_folder = os.path.join(render_data_folder, "log")
    folder_txt = os.path.join(render_log_folder, "folder.txt")
    mesh_txt = os.path.join(render_log_folder, "success.txt")
    # render_folders = read_list(folder_txt)
    thumbnail_op_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../../thumbnail_list.py")
    thumbnail_move_cmd = 'python {} --thumbnail_folder_list {} '.format(
        thumbnail_op_fullpath, folder_txt)
    thumbnail_move_cmd = thumbnail_move_cmd + \
                         ' --mesh_txt_list {} '.format(mesh_txt)
    thumbnail_move_cmd = thumbnail_move_cmd + \
                         ' --output_folder {} '.format(thumbnail_folder)

    print("Copy thumbnail in list file %s to folder %s" %
          (folder_txt, thumbnail_folder))
    os.system(thumbnail_move_cmd)
    time.sleep(0.1)


def generate_upload_threads(data_folder: str,
                            new_upload_path: str,
                            cos_client: CosClient,
                            thread_pool: ThreadPoolExecutor):
    local_file_cos_file_map = {}
    for path, dir_list, file_list in os.walk(data_folder):
        for file_name in file_list:
            current_file_path = os.path.join(path, file_name)
            local_file_key = os.path.relpath(
                current_file_path, data_folder)
            if sys.platform.startswith('win'):
                clean_file_subpath = local_file_key.strip('\\')
                cos_file_subpath = clean_file_subpath.replace('\\', '/')
            else:
                cos_file_subpath = local_file_key.strip('/')

            cos_file_fullpath = new_upload_path + "/" + cos_file_subpath
            print("Upload file from %s (relative path is %s) to path %s" %
                  (current_file_path, local_file_key, cos_file_fullpath))
            thread_pool.submit(cos_client.upload,
                               current_file_path, cos_file_fullpath)
            local_file_cos_file_map[current_file_path] = cos_file_fullpath
    return local_file_cos_file_map


def convert_local_txt_to_remote_txt(local_txt_path: str,
                                    remote_txt_path: str,
                                    local_remote_map: dict):
    success_list = read_list(local_txt_path)
    remote_success_list = []
    for success_element in success_list:
        success_element = success_element.replace("\\", "\\")
        remote_success_list.append(local_remote_map[success_element])
    write_list(remote_txt_path, remote_success_list)


def convert_local_txt_to_remote_list(local_txt_path: str,
                                     local_remote_map: dict):
    success_list = read_list(local_txt_path)
    remote_success_list = []
    for success_element in success_list:
        success_element = success_element.replace("\\", "\\")
        print(success_element)
        remote_success_list.append(local_remote_map[success_element])
    return remote_success_list


def convert_local_mesh_path_to_remote_mesh_path(local_txt_path: str,
                                                local_data_folder: str,
                                                remote_data_folder: str):
    success_list = read_list(local_txt_path)
    remote_success_list = []
    for success_element in success_list:
        print("Beforewards: %s" % (success_element))
        success_element = success_element.replace(
            local_data_folder, remote_data_folder)
        correct_segment = success_element.replace("\\", "/")
        print("Afterwards: %s" % (correct_segment))
        remote_success_list.append(correct_segment)

    return remote_success_list


def convert_local_render_to_remote_render(local_txt_path: str,
                                          local_render_folder: str,
                                          remote_render_folder: str):
    success_list = read_list(local_txt_path)
    remote_success_list = []
    for success_element in success_list:
        print("Beforewards: %s" % (success_element))
        success_element = success_element.replace(
            local_render_folder, remote_render_folder)
        correct_segment = success_element.replace("\\", "/")
        print("Afterwards: %s" % (correct_segment))
        remote_success_list.append(correct_segment)
    return remote_success_list


def scan_files_of_format(data_folder: str, list_filepath: str, folder_list_filepath: str, file_type: str):
    list_generation_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "generate_mesh_list.py")
    scan_max_file_list_cmd = 'python {} --render_folders {} --output_list {} --output_folder_list {} --file_type {}'.format(
        list_generation_fullpath, data_folder, list_filepath, folder_list_filepath, file_type)
    print("Scan for %s files in %s" % (file_type, lycosa_data_folder))
    os.system(scan_max_file_list_cmd)
    time.sleep(0.1)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Windows side lycosa process start. Local time is %s" %
          (local_time_str))

    parser = argparse.ArgumentParser(
        description='Lycosa processing pipeline.')
    parser.add_argument('--lycosa_data_folder', type=str, default="",
                        help='lycosa data folder full path')
    parser.add_argument('--lycosa_data_info_json', type=str, default="",
                        help='lycosa data info json file path')
    parser.add_argument('--data_name', type=str, default="",
                        help='lycosa data name')
    parser.add_argument('--secret_id', type=str,
                        help='secret id of cos account')
    parser.add_argument('--secret_key', type=str,
                        help='secret key of cos account')
    parser.add_argument('--upload_pool_cnt', type=int, default=8,
                        help='upload thread pool ')
    args = parser.parse_args()

    uncompress_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "batch_uncompress_7z.py")
    max_fbx_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "batch_max_fbx.py")
    list_generation_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "generate_mesh_list.py")
    batch_render_fullpath = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "../../render_mesh_batch.py")

    lycosa_data_folder = args.lycosa_data_folder
    lycosa_data_folder = decorate_space(lycosa_data_folder)
    lycosa_data_info_json = args.lycosa_data_info_json
    data_name = args.data_name
    upload_pool_cnt = args.upload_pool_cnt

    cos_client = CosClient(args.secret_id, args.secret_key)

    max_convert_log_folder = os.path.join(
        lycosa_data_folder, "max_convert_log")
    if not os.path.exists(max_convert_log_folder):
        os.mkdir(max_convert_log_folder)

    uncompress_cmd = 'python \"{}\" --decompress_folder \"{}\"'.format(
        uncompress_fullpath, lycosa_data_folder)
    print("Uncompressing files in %s" % (lycosa_data_folder))
    os.system(uncompress_cmd)
    time.sleep(0.1)

    max_list_filepath = os.path.join(lycosa_data_folder, "max_list.txt")
    max_folder_list_filepath = os.path.join(
        lycosa_data_folder, "max_folder_list.txt")

    scan_files_of_format(data_folder=lycosa_data_folder,
                         list_filepath=max_list_filepath,
                         folder_list_filepath=max_folder_list_filepath,
                         file_type=".max")

    if os.path.exists(max_list_filepath) and os.path.exists(max_folder_list_filepath):
        max_file_list = read_list(max_list_filepath)
        max_folder_list = read_list(max_folder_list_filepath)

        max_fbx_cmd = "python \"{}\" --max_file_list \"{}\" --max_folder_list \"{}\" --log_folder \"{}\"".format(
            max_fbx_fullpath, max_list_filepath, max_folder_list_filepath, max_convert_log_folder)

        print("Convert 3dsmax files to fbx format....")
        os.system(max_fbx_cmd)
        time.sleep(0.1)

    to_hash_str = lycosa_data_folder + data_name + local_time_str
    hash_obj = hashlib.sha1(to_hash_str.encode('utf-8'))
    hash_foldername = str(hash_obj.hexdigest())
    # hash_foldername=data_name+"_"+local_time_str+hash_foldername
    new_upload_path = "Asset/lycosa/" + hash_foldername + "/" + data_name
    # cos_client.create_folder(new_upload_path)

    model_upload_pool = ThreadPoolExecutor(max_workers=upload_pool_cnt,
                                           thread_name_prefix='lycosa_upload')
    model_local_cos_map = generate_upload_threads(data_folder=lycosa_data_folder,
                                                  new_upload_path=new_upload_path,
                                                  cos_client=cos_client,
                                                  thread_pool=model_upload_pool)
    time.sleep(10)
    model_upload_pool.shutdown()
    time.sleep(10)

    lycosa_data_info = {}
    lycosa_data_info["remote_folder"] = "/mnt/aigc_bucket_1/" + new_upload_path

    write_json(lycosa_data_info_json, lycosa_data_info)

    t_end = time.time()
    local_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Windows side lycosa process end. Start time is %s; end time is %s" %
          (local_time_str, end_time_str))
