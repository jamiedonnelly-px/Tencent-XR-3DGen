import time

import pulsar
from _pulsar import ConsumerType

"""
灾备模式：
consumer 将会按字典顺序排序，第一个 consumer 被初始化为唯一接受消息的消费者。
当 master consumer 断开时，所有的消息（未被确认和后续进入的）将会被分发给队列中的下一个 consumer。
"""


# 消费监听
def listener(consumer, msg):
    # 模拟业务
    print("Received message '{}' id='{}'".format(msg.data(), msg.message_id()))
    # 回复ack
    consumer.acknowledge(msg)


# 创建客户端
client = pulsar.Client(
    authentication=pulsar.AuthenticationToken(
        # 已授权角色密钥
        "eyJrZXlJZC......"),
    # 服务接入地址
    service_url='http://pulsar-xxx.tdmq-pulsar.ap-sh.public.tencenttdmq.com:8080')

# 订阅消息
simpleConsumer = client.subscribe(
    # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
    topic='pulsar-xxx/sdk_python/topic1',
    # 订阅名称
    subscription_name='sub_topic1',
    # 设置监听
    message_listener=listener,
    # 设置订阅模式为 Failover（灾备）模式
    consumer_type=ConsumerType.Failover
)

while True:
    time.sleep(1000)
