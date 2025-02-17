#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
from polaris.api.consumer import *
from polaris.pkg.model.error import *
from polaris.wrapper import POLARIS_CALL_RET_OK, POLARIS_CALL_RET_ERROR

# consumer api 内部有缓存，全局只使用一个consumer api对象
g_consumer_api = None


def get_consumer_api():
    global g_consumer_api
    if not g_consumer_api:
        g_consumer_api = create_consumer_by_default_config_file()
    return g_consumer_api


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("python3 main.py namespace service_name times , example:python test_polaris.py Development gdp.xr3d.tricup-high-dev-v3 10")
        sys.exit(1)
    namespace = sys.argv[1]
    service = sys.argv[2]
    times = int(sys.argv[3])

    for i in range(times):
        # 业务程序中，针对每个请求均进行1次服务发现获取被调地址
        request = GetOneInstanceRequest(namespace=namespace, service=service)
        # 1. 获取被调服务的一个实例
        try:
            instance = get_consumer_api().get_one_instance(request)
        except SDKError as e:
            raise Exception("Polaris SDK error %s" % repr(e))

        # 2. 业务进行实际的RPC调用，例子只输出被调IP和PORT
        print(f'time:{i} ' + 'host:{host} port:{port}'.format(host=instance.get_host(), port=instance.get_port()))
        # 计算RPC返回码和延迟
        ret_code = 0  # 赋值实际的RPC调用结果
        delay = 100  # 业务计算实际RPC结果
        time.sleep(0.3)

        # 3. RPC结果后执行调用结果上报
        service_call_result = ServiceCallResult(namespace=namespace, service=service, instance_id=instance.get_id())
        service_call_result.set_delay(delay)
        service_call_result.set_ret_code(ret_code)
        # 注意：根据实际的返回计算调用结果：
        #  - POLARIS_CALL_RET_OK表示调用成功，建议业务错误上报成RPC成功，避免因为业务错误导致剔除被调
        #  - POLARIS_CALL_RET_ERROR表示调用失败，用于熔断剔除，建议网络失败上报成POLARIS_CALL_RET_ERROR
        ret_status = POLARIS_CALL_RET_OK if ret_code >= 0 else POLARIS_CALL_RET_ERROR
        service_call_result.set_ret_status(ret_status)
        get_consumer_api().update_service_call_result(service_call_result)

#     for i in range(times):
#         # 通过服务名获取全部服务实例
#         request = GetInstancesRequest(namespace=namespace, service=service)
#         # 1. 调用获取服务全部实例的接口
#         try:
#             response = get_consumer_api().get_all_instances(request)
#         except SDKError as e:
#             raise Exception("Polaris SDK error %s" % repr(e))

#         # 2. 轮询服务实例列表
#         count = 0
#         for instance in response:
#             print('host:{host} port:{port}'.format(host=instance.get_host(), port=instance.get_port()))
#             count += 1
#         print('response size:{size}, count:{count}'.format(size=response.size(), count=count))

#         time.sleep(1)        


## all ip
request = GetInstancesRequest(namespace=namespace, service=service)
response = get_consumer_api().get_all_instances(request)
server_ips = [f'host:{instance.get_host()} port:{instance.get_port()}' for instance in response]
jobs = range(10)    # N 个客户端任务, 省略线程池实现

def run_one_job(job, max_query_cnt = 10):
    run_result = None
    for i in range(max_query_cnt):
        request = GetOneInstanceRequest(namespace=namespace, service=service)
        # 1. 获取被调服务的一个实例
        try:
            instance = get_consumer_api().get_one_instance(request)
        except SDKError as e:
            raise Exception("Polaris SDK error %s" % repr(e))
        once_ip = f'host:{instance.get_host()} port:{instance.get_port()}'
        
        # 2. 查询ip对应的服务的状态
        is_busy = grpc_query_state(once_ip)    # grpc查询对应服务的状态
        if is_busy:
            continue
        else:
            # 3. 发送grpc服务请求
            run_result = grpc_send_request(once_ip, job)  # 如果空闲, 发送grpc请求, run ~10s
            print(f"Complete the task after {i+1} attempts")
            break       
    return run_result

# use all mode     
for server_ip in server_ips:
    is_busy = query_state(server_ip)    # 查询对应服务的状态
    if is_busy:
        continue
    else:
        run_result = send_request(server_ip, select_and_pop_job(jobs))  # 如果空闲, 发送请求, 怎么异步?