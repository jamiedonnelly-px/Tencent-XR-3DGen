import os
import time
import logging
import subprocess

need_check_dir = '/mnt/aigc_bucket_4/sz/mount_ok.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_and_kill_python_processes():
    # 检查文件夹是否存在
    if not os.path.exists(need_check_dir):
        logging.error(f"Directory {need_check_dir} does not exist. Killing all Python processes...")
        # 使用subprocess查找并终止所有Python进程
        try:
            # 在Unix/Linux系统中
            subprocess.run(['pkill', '-f', 'service_side.py'])
            os.system("pkill blender")
            # subprocess.run(['pkill', 'blender'], check=True)
            logging.error(f"kill all done...")
            return False
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while trying to kill Python processes: {e}")
    else:
        logging.info(f"Directory {need_check_dir} exists.")
    
    return True

flag = True
while flag:
    flag = check_and_kill_python_processes()
    time.sleep(10)  # 每隔10秒检查一次

logging.error(f"ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")