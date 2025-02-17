import argparse
import os
import time

import yaml


def read_list(in_list_txt: str):
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        exit(-1)

    str_list = []
    with open(in_list_txt, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            str_list.append(line.strip())
    return str_list


def write_list(path: str, write_list: list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def parse_lanjing_pods(name, kubeconfig: str = ''):
    log_file = '/tmp/lanjing_pods.txt'
    cmd = "kubectl get pod -n ieg-aigc3d-4-game-gpu-nj --kubeconfig {} | grep Running > {}".format(kubeconfig, log_file)
    os.system(cmd)

    pod_name_list = []
    pod_info_list = read_list(log_file)
    for pod_info in pod_info_list:
        pod_fullname = pod_info.split()[1]
        pod_name_elements = pod_fullname.split("-")[:-2]
        pod_name = '_'.join(pod_name_elements)
        if pod_name == name:
            pod_name_list.append(pod_fullname)

    return pod_name_list


def parse_pods_from_yaml(yaml_path, kubeconfig_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        name = yaml_data['metadata']['name']
        gpu_cnt = yaml_data['spec']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu']
    print("Pod name is %s; kube config file locates at %s...." % (name, kubeconfig_path))
    pods = parse_lanjing_pods(name, kubeconfig_path)
    return pods


def decorate_cmd(job_str: str):
    input_part = "pod_num=$1 \npod_id=$2 \ncode_folder=$3 \n"
    tmux_cmd = "tmux kill-session -t render \ntmux new -t render -d \ntmux send -t render \"bash {} $pod_num $pod_id $code_folder\" C-m".format(
        job_str)
    total_command = input_part + tmux_cmd
    return total_command


def start_pods(pod_cmd_folder: str, yaml_path: str,
               kubeconfig_path: str, script_path: str):
    pods = parse_pods_from_yaml(yaml_path, kubeconfig_path)
    pods_cnt = len(pods)

    for pod_id in range(pods_cnt):
        pod_name = pods[pod_id]

        print("No %i pod from in total %i pods; name is %s" % (pod_id, pods_cnt, pod_name))

        mnt_cmd = "bash  /workspace/mount.sh"
        mnt_lj_cmd = "kubectl exec -it {} -n ieg-aigc3d-4-game-gpu-nj --kubeconfig {} {}".format(
            pod_name, kubeconfig_path, mnt_cmd)
        os.system(mnt_lj_cmd)
        time.sleep(0.1)

        current_code_folder = os.path.dirname(os.path.abspath(__file__))

        lanjing_shell = decorate_cmd(script_path)
        lanjing_script_path = os.path.join(pod_cmd_folder, pod_name + ".sh")
        with open(lanjing_script_path, 'w') as f:
            f.write(lanjing_shell)
        final_shell_cmd = "'bash \'{}\' {} {} {}'".format(
            lanjing_script_path, pods_cnt, pod_id, current_code_folder)
        time.sleep(0.1)

        cmd = "kubectl exec -it {} -n ieg-aigc3d-4-game-gpu-nj --kubeconfig {} -- bash -c {}".format(
            pod_name, kubeconfig_path, final_shell_cmd)
        print('Execution CMD: ', cmd)
        os.system(cmd)
        time.sleep(0.1)


if __name__ == "__main__":
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Distribute jobs start. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(description='Generate data repo json')
    parser.add_argument('--pod_cmd_folder', type=str, default="",
                        help='json containing commands for each pod')
    parser.add_argument('--pod_script_path', type=str, default="",
                        help='script for running distributed')
    parser.add_argument('--kubeconfig_path', type=str, default="",
                        help='lanjing k8s cluster Auth file path')
    parser.add_argument('--pod_yaml_path', type=str, default="",
                        help='lanjing k8s cluster name and information file path')
    args = parser.parse_args()

    if not os.path.exists(args.pod_cmd_folder):
        os.mkdir(args.pod_cmd_folder)

    start_pods(args.pod_cmd_folder, args.pod_yaml_path,
               args.kubeconfig_path, args.pod_script_path)
