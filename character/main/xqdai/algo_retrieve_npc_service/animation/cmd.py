import os
import subprocess

def run_command(cmd):
    try:
        # 使用 subprocess.run 执行命令
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Output:", result.returncode)
        return 0
    except:
        return -1

def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension

# cmd = "/root/blender-3.6.15-linux-x64/blender   --background --addons rokoko-studio-live-blender-master  --python  /mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/animation/retarget_test.py  --  1.fbx 2.fbx 3.fbx"
# ret = run_command(cmd)
# print(ret)
print(get_file_extension("/mnt/aigc_cfs_cq/xiaqiangdai/project/algo_retrieve_npc_service/animation/010447.fbx"))