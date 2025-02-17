import pulsar

from pulsar import MessageId

# 创建客户端
client = pulsar.Client(
    authentication=pulsar.AuthenticationToken(
        # 已授权角色密钥
        "eyJrZXlJZC......"),
    # 服务接入地址
    service_url='http://pulsar-xxx.tdmq-pulsar.ap-sh.public.tencenttdmq.com:8080')

# 创建reader订阅
reader = client.create_reader(
    # 使用 Reader 方式订阅主题时，需要指定到 Topic 分区级别（默认分区即在 Topic 后面加 -partition-0）
    topic="pulsar-xxx/sdk_python/topic1-partition-0",
    # 设置从起始位置开始
    start_message_id=MessageId.earliest
)

while True:
    # 获取消息
    msg = reader.read_next()
    # 模拟业务
    print("Received message '{}' id='{}'".format(msg.data(), msg.message_id()))
