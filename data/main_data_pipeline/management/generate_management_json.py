import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client


def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate))
        sys.stdout.flush()


class CosClient():
    def __init__(self, secret_id: str, secret_key: str):
        region = 'ap-nanjing'
        domain = '3d-aigc-fast-1258344700.cos-internal.ap-nanjing.tencentcos.cn.'
        self.bucket = '3d-aigc-fast-1258344700'  # 存储桶名称
        token = None  # 使用临时密钥时，需要传入Token，默认为空
        # scheme = 'http'  #
        scheme = 'https'  # 指定使用 http/https 协议来访问 COS，默认为 https，可不填
        # 使用内网域名接入if
        endpoint = 'cos-internal.%s.tencentcos.cn' % region  # 桶域名(访问具体桶时使用)
        # 服务域名(列举所有桶时使用)(此域名暂时不支持内网域名，这里仍然使用的外网域名)
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
            progress_callback=percentage
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


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def read_list(in_list_txt):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path, write_list):
    with open(path, 'w', encoding='utf-8') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def find_match_between_mesh_name_and_thumbnail(mesh_path: str, thumbnail_set: set):
    folder = os.path.split(mesh_path)[0]
    filename = os.path.split(mesh_path)[1]
    file_basename = os.path.splitext(filename)[0]

    original_elements = folder.split("/")
    original_elements.append(file_basename)
    elements = []
    for element in original_elements:
        new_element = element.replace(' ', '__')
        elements.append(new_element)

    mesh_element_number = len(elements)
    for index in range(mesh_element_number - 1, -1, -1):
        connect_name_str = "_".join(elements[index:mesh_element_number])
        if connect_name_str in thumbnail_set:
            return connect_name_str
    return None


def scan_thumbnail_names(thumbnail_folder: str, thumbnail_name_path_map: dict):
    if not os.path.exists(thumbnail_folder):
        print("Cannot find thumbnail in folder %s" % (thumbnail_folder))
        return False

    thumbnail_name_path_map = {}
    for root, dirs, files in os.walk(thumbnail_folder):
        for dir in dirs:
            current_folder_fullpath = os.path.join(root, dir)
            print("Check mesh in folder %s" % (current_folder_fullpath))
            thumbnail_image_path = os.path.join(
                current_folder_fullpath, "thumbnail.png")
            if os.path.exists(thumbnail_image_path):
                thumbnail_name_path_map[str(dir)] = thumbnail_image_path

    return True


def fill_dummy_in_management_json(json_struct):
    json_struct["ScrapyUrl"] = ""
    json_struct["Comment"] = ""


def manage_once(mesh_path: str,
                thumbnail_folder: str,
                mesh_basic_script: str,
                everything_fbx_converter: str,
                data_info_struct: dict,
                remote_fbx: str,
                remote_thumbnail: str,
                stat_txt: str,
                folder_txt: str,
                layer: bool):
    stat = 0
    t_start = time.time()
    start_time = time.localtime(t_start)
    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', start_time)

    print('Start time for management generation cmd is %s....' %
          (str(start_time_str)))

    mesh_data = {}
    mesh_filename = os.path.split(mesh_path)[1]
    mesh_basename = os.path.splitext(mesh_filename)[0]

    thumbnail_png_path = os.path.join(thumbnail_folder, "thumbnail.png")
    if not os.path.exists(thumbnail_png_path):
        return

    mesh_info_json_path = os.path.join(thumbnail_folder, mesh_basename + ".json")
    if not os.path.exists(mesh_info_json_path):
        mesh_basic_cmd = "{} -b -P \"{}\" -- --mesh_path \"{}\" --output_mesh_json_info \"{}\" ".format(
            blender_root, mesh_basic_script, mesh_path, mesh_info_json_path)

        print(mesh_basic_cmd)
        os.system(mesh_basic_cmd)
        time.sleep(0.1)

    if not os.path.exists(mesh_info_json_path):
        return

    mesh_basic_info = read_json(mesh_info_json_path)

    thumbnail_parent_folder = os.path.split(thumbnail_folder)[0]
    thumbnail_category_folder = os.path.split(thumbnail_parent_folder)[0]
    key_name = os.path.split(thumbnail_parent_folder)[1]
    category_name = os.path.split(thumbnail_category_folder)[1]

    print("Mesh category is %s, key is %s, info is %s..." %
          (category_name, key_name, str(mesh_basic_info)))

    mesh_data["Key"] = key_name
    mesh_data["Category"] = category_name
    mesh_data["MeshName"] = mesh_filename

    print(data_info_struct)

    if data_info_struct is not None:
        if category_name not in data_info_struct["data"].keys():
            return
        if key_name not in data_info_struct["data"][category_name].keys():
            return

        render_folder = data_info_struct["data"][category_name][key_name]["ImgDir"]
        if "GeoPcd" in data_info_struct["data"][category_name][key_name].keys():
            geometry_folder = data_info_struct["data"][category_name][key_name]["GeoPcd"]
        else:
            geometry_folder = None
        if "TexPcd" in data_info_struct["data"][category_name][key_name].keys():
            texture_folder = data_info_struct["data"][category_name][key_name]["TexPcd"]
        else:
            texture_folder = None
    else:
        render_folder = None
        geometry_folder = None
        texture_folder = None

    pack_fbx_path = os.path.join(thumbnail_folder, key_name + ".fbx")
    if not os.path.exists(pack_fbx_path):
        fbx_convert_cmd = "{} -b -P \"{}\" -- --mesh_path \"{}\" --output_fullpath \"{}\" ".format(
            blender_root, everything_fbx_converter, mesh_path, pack_fbx_path)
        fbx_convert_cmd = fbx_convert_cmd + " --pack "

        print(fbx_convert_cmd)
        os.system(fbx_convert_cmd)
        time.sleep(0.1)

    remote_fbx_filename, fbx_url = get_remote_thumbnail_name(
        cos_client, pack_fbx_path, remote_fbx)
    remote_thumbnail_filename, thumbnail_url = get_remote_thumbnail_name(
        cos_client, thumbnail_png_path, remote_thumbnail)

    real_remote_thumbnail_filename = remote_thumbnail_filename[1:]
    real_remote_fbx_filename = remote_fbx_filename[1:]

    mesh_data["SavePaths"] = {}
    mesh_data["SavePaths"]["MeshThumbnailFilename"] = real_remote_thumbnail_filename
    mesh_data["SavePaths"]["MeshFBXFilename"] = real_remote_fbx_filename
    mesh_data["SavePaths"]["MeshRenderFolder"] = render_folder
    mesh_data["SavePaths"]["MeshGeometryFolder"] = geometry_folder
    mesh_data["SavePaths"]["MeshTextureFolder"] = texture_folder
    mesh_data["SavePaths"]["MeshFilename"] = mesh_path

    mesh_data["GameCategory"] = data_name
    mesh_data["MeshStyle"] = data_style
    mesh_data["Origin"] = data_origin
    mesh_data["index"] = -1
    mesh_data["FaceNum"] = mesh_basic_info["FaceNum"]

    mesh_data["IfPropertiesExist"] = {}
    mesh_data["IfPropertiesExist"]["TextureExist"] = mesh_basic_info["TextureExist"]
    mesh_data["IfPropertiesExist"]["RoughnessExist"] = mesh_basic_info["RoughnessExist"]
    mesh_data["IfPropertiesExist"]["MetallicExist"] = mesh_basic_info["MetallicExist"]
    mesh_data["IfPropertiesExist"]["SpecularExist"] = mesh_basic_info["SpecularExist"]
    mesh_data["IfPropertiesExist"]["NormalExist"] = mesh_basic_info["NormalExist"]

    mesh_data["Specific"] = {}
    mesh_data["Specific"]["Layer"] = layer

    fill_dummy_in_management_json(mesh_data)

    mesh_data_json_str = json.dumps(mesh_data)

    t_end = time.time()
    end_time = time.localtime(t_end)
    end_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', end_time)

    print('After management generation cmd status is %s; time for this status is %s....' % (
        str(stat), str(end_time_str)))

    with open(stat_txt, 'a') as f:
        f.write('{}\n'.format(mesh_path))

    with open(folder_txt, 'a') as f:
        f.write('{}\n'.format(mesh_data_json_str))


def get_remote_thumbnail_name(cos_client: CosClient, local_thumbnail_path: str, remote_thumbnail_folder: str):
    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)

    thumbnail_type = os.path.splitext(local_thumbnail_path)[1]

    to_hash_str = local_thumbnail_path + "-" + local_time_str
    hash_obj = hashlib.sha1(to_hash_str.encode('utf-8'))
    hash_thumbnail_name = str(hash_obj.hexdigest()) + thumbnail_type

    remote_path = os.path.join(remote_thumbnail_folder, hash_thumbnail_name)

    thumbnail_url = cos_client.upload(
        local_path=local_thumbnail_path, remote_path=remote_path)

    return remote_path, thumbnail_url


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate management system json')
    parser.add_argument('--mesh_list', type=str,
                        help='list txt file containing mesh abs path')
    parser.add_argument('--folder_list', type=str,
                        help='list txt file containing rendered mesh output folder')
    parser.add_argument('--data_json', type=str, default='',
                        help='data json file for storing data info')
    parser.add_argument('--data_name', type=str,
                        help='data from which game/website; data name in final json script (e.g. daz/vroid/hok....)')
    parser.add_argument('--data_style', type=str, default='ACG',
                        help='data style, choose between DAZ/Voxel/ACG/Stylized/PR')
    parser.add_argument('--data_origin', type=str, default='',
                        help='origin of data (downloaded/collected from which place)')
    parser.add_argument('--output_json', type=str,
                        help='output management system json path')
    # parser.add_argument('--output_thumbnail_json', type=str,
    #                     help='output mesh path thumbnail filename json path')
    parser.add_argument('--secret_id', type=str,
                        help='secret id of cos account')
    parser.add_argument('--secret_key', type=str,
                        help='secret key of cos account')
    parser.add_argument('--pool_cnt', type=int, default=8,
                        help='multiprocessing pool cnt')
    parser.add_argument('--log_folder', type=str, default='./log',
                        help='log folder to store information')
    parser.add_argument('--blender_root', type=str,
                        default='/root/blender-3.6.2-linux-x64/blender', help='path for blender binary exe')
    parser.add_argument('--layer', action='store_true',
                        help='marks if this data is layer data')

    args = parser.parse_args()
    blender_root = args.blender_root
    pool_cnt = args.pool_cnt
    log_folder = args.log_folder
    mesh_list_txt = args.mesh_list
    folder_list_txt = args.folder_list
    data_json = args.data_json
    data_name = args.data_name
    data_style = args.data_style
    data_origin = args.data_origin
    output_json = args.output_json
    layer = args.layer
    # output_thumbnail_json = args.output_thumbnail_json

    cos_client = CosClient(args.secret_id, args.secret_key)
    original_mesh_path_list = read_list(mesh_list_txt)
    original_thumbnail_folder_list = read_list(folder_list_txt)
    if len(data_json) > 0:
        data_struct = read_json(data_json)
    else:
        data_struct = None

    mesh_path_list = []
    thumbnail_folder_list = []
    mesh_path_set = set(original_mesh_path_list)
    for index in range(len(original_thumbnail_folder_list)):
        thumbnail_folder = original_thumbnail_folder_list[index]
        thumbnail_mesh_path = original_mesh_path_list[index]

        thumbnail_parent_folder = os.path.split(thumbnail_folder)[0]
        thumbnail_category_folder = os.path.split(thumbnail_parent_folder)[0]
        key_name = os.path.split(thumbnail_parent_folder)[1]
        category_name = os.path.split(thumbnail_category_folder)[1]
        if data_struct is not None:
            if category_name in data_struct["data"].keys():
                if key_name in data_struct["data"][category_name].keys():
                    current_mesh_path = data_struct["data"][category_name][key_name]["Mesh"]
                    if current_mesh_path not in mesh_path_set:
                        continue
                    mesh_path_list.append(current_mesh_path)
                    thumbnail_folder_list.append(thumbnail_folder)
        else:
            mesh_path_list.append(thumbnail_mesh_path)
            thumbnail_folder_list.append(thumbnail_folder)

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    cpu_cnt = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.pool_cnt)
    print('Find {} cpus, use {} threads in converting......'.format(
        cpu_cnt, args.pool_cnt))

    stat_txt = os.path.join(log_folder, 'success.txt')
    folder_txt = os.path.join(log_folder, 'folder.txt')
    stat_file = open(stat_txt, 'w')
    folder_file = open(folder_txt, 'w')

    mesh_basic_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../manifold/mesh_basic.py")
    everything_fbx_converter = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "../conversion/anything_fbx_converter.py")

    remote_thumbnail = "/data/thumbnails/lycosa"
    remote_fbx = "/data/fbx"

    for index in range(len(mesh_path_list)):
        mesh_path = mesh_path_list[index]
        thumbnail_folder = thumbnail_folder_list[index]

        pool.apply_async(func=manage_once, args=(mesh_path, thumbnail_folder, mesh_basic_script,
                                                 everything_fbx_converter, data_struct, remote_fbx,
                                                 remote_thumbnail, stat_txt, folder_txt, layer))

    pool.close()
    pool.join()
    time.sleep(0.1)

    stat_file.close()
    folder_file.close()
    time.sleep(0.1)

    management_struct = {}
    management_struct["data"] = []
    management_list = read_list(folder_txt)
    for management_info in management_list:
        current_json_object = json.loads(management_info)
        management_struct["data"].append(current_json_object)

    write_json(output_json, management_struct)
