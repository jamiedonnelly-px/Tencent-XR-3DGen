import time

import pulsar
from _pulsar import ConsumerType

"""
共享模式:
消息通过 round robin 轮询机制（也可以自定义）分发给不同的消费者，并且每个消息仅会被分发给一个消费者。
当消费者断开连接，所有被发送给他，但没有被确认的消息将被重新安排，分发给其它存活的消费者。
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
    # 设置订阅模式为 Shared（共享）模式
    consumer_type=ConsumerType.Shared
)

while True:
    time.sleep(1000)
