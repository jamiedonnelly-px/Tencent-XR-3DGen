import pulsar

# 创建客户端
client = pulsar.Client(
    authentication=pulsar.AuthenticationToken(
        # 已授权角色密钥
        "eyJrZXlJZC......"),
    # 服务接入地址
    service_url='http://pulsar-xxx.tdmq-pulsar.ap-sh.public.tencenttdmq.com:8080')

# 订阅消息
consumer = client.subscribe(
    # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
    topic='pulsar-xxx/sdk_python/topic1',
    # 订阅名称
    subscription_name='sub_topic1'
)

while True:
    # 获取消息
    msg = consumer.receive()
    try:
        # 模拟业务
        print(
            "Received message '{}' id='{}'".format(msg.data(), msg.message_id()))
        # 消费成功，回复ack
        consumer.acknowledge(msg)
    except:
        # 消费失败，消息将会重新投递
        consumer.negative_acknowledge(msg)
