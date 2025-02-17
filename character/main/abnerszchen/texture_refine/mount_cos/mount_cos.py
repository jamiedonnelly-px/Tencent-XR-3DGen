import os
import argparse
import time
import xml.etree.ElementTree as ET
import shutil


AREA_SUFFIX = {"cq": "cos-internal.ap-chongqing.tencentcos.cn",
               "nj": "cos-internal.ap-nanjing.tencentcos.cn",
               "gz": "cos-internal.ap-guangzhou.tencentcos.cn",
               "nj_fast": "cos-internal.ap-nanjing.tencentcos.cn"}


AREA_BUCKET = {"cq": "3d-aigc-1258344700",
               "nj": "3d-agic-nanjing-1258344700",
               "gz": "3d-aigc-guangzhou2-1258344700",
               "nj_fast": "3d-aigc-fast-1258344700"}

AREA_MOUNT = {"cq": "/mnt/aigc_bucket_1",
              "nj": "/mnt/aigc_bucket_2",
              "gz": "/mnt/aigc_bucket_3",
              "nj_fast" : "/mnt/aigc_bucket_4"}


def generate_xml_config(xml_path: str, output_xml_path: str, secretId: str, secretKey: str, area: str):
    if not os.path.exists(xml_path):
        print("Cannot find xml config at %s; specify abs path of core-site.xml in --xml_path when using the script" % (xml_path))
        return

    tree = ET.parse(xml_path)
    for property_node in tree.getroot().iter("property"):
        property_name = property_node.find("name").text
        property_value = property_node.find("value").text

        if property_name == "fs.cosn.userinfo.secretId":
            property_node.find("value").text=(secretId)
        if property_name == "fs.cosn.userinfo.secretKey":
            property_node.find("value").text=(secretKey)
        if property_name == "fs.cosn.bucket.endpoint_suffix":
            if area in AREA_SUFFIX:
                property_node.find("value").text=(AREA_SUFFIX[area])
    
    tree.write(output_xml_path)


def read_list(in_list_txt, to_lower=False):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                if to_lower:
                    str_list.append(mesh_path.lower())
                else:
                    str_list.append(mesh_path)
    return str_list


def check_str_in_list(check_str: str, str_list: list):
    for str_element in str_list:
        if check_str in str_element:
            return str_list.index(str_element)
    return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch download files to mesh folder')
    parser.add_argument('--secret_id', type=str,
                        help='secret id of your cos')
    parser.add_argument('--secret_key', type=str,
                        help='secret key of your cos')
    # parser.add_argument('--area', type=str,
    #                     help='area of cos bucket, choose from cq/nj/gz')
    parser.add_argument('--xml_path', type=str, default="core-site.xml",
                        help='cos config xml path')
    parser.add_argument("--skip_install", action="store_true", default=False,help="skip install when run in docker")
    parser.add_argument("--only_install", action="store_true", default=False,help="only install for build docker")
    args = parser.parse_args()

    if not args.skip_install:
        os_info = read_list("/etc/os-release", to_lower=True)
        print("OS information is:")
        print(os_info)
        line_index = check_str_in_list("version", os_info)

        print("Install dependecy....")

        if "tencentos" in os_info[0]:
            if "3.2" in os_info[line_index] or "3.1" in os_info[line_index]:
                cmd = "yum install -y java-11-konajdk fuse-devel gcc curl  > /dev/null"
                os.system(cmd)
            # elif "2.6" in os_info[line_index]:
            #     cmd = "yum install tlinux-release-tencentkona -y && yum install -y java-11-konajdk fuse-devel gcc curl"
            #     os.system(cmd)
            else:
                print("OS version error: version is %s" % (os_info[line_index]))
                exit(-1)
        elif "tlinux" in os_info[0]:
            if "3.2" in os_info[line_index] or "3.1" in os_info[line_index]:
                cmd = "yum install -y java-11-konajdk fuse-devel gcc curl  > /dev/null"
                os.system(cmd)
            # elif "2.6" in os_info[line_index]:
            #     cmd = "yum install tlinux-release-tencentkona -y && yum install -y java-11-konajdk fuse-devel gcc curl"
            #     os.system(cmd)
            else:
                print("OS version error: version is %s" % (os_info[line_index]))
                exit(-1)
        else:
            cmd = "apt-get update  > /dev/null && apt-get install -y libfuse-dev curl build-essential default-jdk  > /dev/null"
            os.system(cmd)

        time.sleep(0.1)


        print("Download and install goosefs-lite....")
        cmd = "curl -LO \'https://downloads.tencentgoosefs.cn/goosefs-lite/goosefs-lite-1.0.3.tar.gz\' > /dev/null"
        os.system(cmd)
        time.sleep(0.1)

        cmd = "tar -xvf goosefs-lite-1.0.3.tar.gz > /dev/null"
        os.system(cmd)
        time.sleep(0.1)

        cmd = "cd goosefs-lite-1.0.3 && bash bin/init.sh"
        os.system(cmd)
        time.sleep(0.1)

        cmd = "export JAVA_OPTS=\" -Xms16G -Xmx16G  -XX:MaxDirectMemorySize=16G -XX:+UseG1GC\"  > /dev/null"
        os.system(cmd)
        time.sleep(0.1)
    
    if not args.only_install:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        original_config = os.path.join(
            current_dir, "goosefs-lite-1.0.3/conf/core-site.xml")
        backup_config = os.path.join(
            current_dir, "goosefs-lite-1.0.3/conf/core-site-bak.xml")

        if os.path.exists(original_config):
            shutil.move(original_config, backup_config)
            for area in AREA_SUFFIX.keys():
                generate_xml_config(args.xml_path, original_config,
                                    args.secret_id, args.secret_key, area)

                cmd = "mkdir -p {}".format(AREA_MOUNT[area])
                os.system(cmd)
                time.sleep(0.1)

                mount_point_file_list=os.listdir(AREA_MOUNT[area])
                if len(mount_point_file_list)>0:
                    continue

                cmd = "cd ./goosefs-lite-1.0.3/ && ./bin/goosefs-lite mount {} cosn://{}/".format(
                    AREA_MOUNT[area], AREA_BUCKET[area])
                os.system(cmd)
                time.sleep(0.1)

                cmd = "ps -ef|grep goosefs-lite|grep -v grep "
                os.system(cmd)
                time.sleep(0.1)
                print(f"mount cos {area} ok")
