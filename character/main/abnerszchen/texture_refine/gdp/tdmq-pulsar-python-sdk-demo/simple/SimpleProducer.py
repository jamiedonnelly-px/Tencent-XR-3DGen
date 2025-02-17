import pulsar

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

for i in range(10):
    # 发送消息
    producer.send(
        # 消息内容
        ('Hello python client, this is a msg-%d' % i).encode('utf-8'),
        # 消息参数
        properties={'k': 'v'},
        # 业务key
        partition_key='yourKey' + str(i % 3)
    )

client.close()
