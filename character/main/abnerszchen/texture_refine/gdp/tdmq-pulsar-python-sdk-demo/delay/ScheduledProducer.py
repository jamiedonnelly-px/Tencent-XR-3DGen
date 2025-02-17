import time

import pulsar

"""
定时消息需要使用shared模式的消费才有效果，其他订阅模式会失去效果
"""

# 创建客户端
client = pulsar.Client(
    authentication=pulsar.AuthenticationToken(
        # 已授权角色密钥
        "eyJrZXlJZC......"),
    # 服务接入地址
    service_url='http://pulsar-xxx.tdmq-pulsar.ap-sh.public.tencenttdmq.com:8080')

# 创建生产者
producer = client.create_producer(
    # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
    topic='pulsar-xxx/sdk_python/topic1'
)

# 发送消息
producer.send(
    # 消息内容
    'Hello python client, this is a scheduled msg'.encode('utf-8'),
    # 消息参数
    properties={'k': 'v'},
    # 业务key
    partition_key='yourKey',
    # 设置发送时间戳
    deliver_at=int(round(time.time() * 1000)) + 10000
)

client.close()
