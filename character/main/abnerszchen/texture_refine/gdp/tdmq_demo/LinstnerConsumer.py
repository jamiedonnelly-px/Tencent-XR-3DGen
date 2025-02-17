import time
import pulsar

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
        "token"),
    # 服务接入地址
    service_url='http://pulsar-32j8nb4k7393.eap-ds7yxa7i.tdmq.ap-nj.internal.tencenttdmq.com:8080')

# 订阅消息
simpleConsumer = client.subscribe(
    # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
    topic='persistent://pulsar-32j8nb4k7393/aigc-x-1/aigc-topic-1st',
    # 订阅名称
    subscription_name='sub_topic1',
    # 设置监听
    message_listener=listener
)

while True:
    time.sleep(1000)
