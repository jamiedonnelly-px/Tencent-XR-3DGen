import pulsar
import time

# 异步发送回调
def send_callback(send_result, msg_id):
    print('Message published: result:{}  msg_id:{}'.format(send_result, msg_id))


# 创建客户端
client = pulsar.Client(
    authentication=pulsar.AuthenticationToken(
        # 已授权角色密钥
        "token"),
    # 服务接入地址
    service_url='http://pulsar-32j8nb4k7393.eap-ds7yxa7i.tdmq.ap-nj.internal.tencenttdmq.com:8080')

# 创建生产者
producer = client.create_producer(
    # topic完整路径，格式为persistent://集群（租户）ID/命名空间/Topic名称，从【Topic管理】处复制
    topic='persistent://pulsar-32j8nb4k7393/aigc-x-1/aigc-topic-zero123plus'
)

for i in range(10):
    # 发送消息
    producer.send_async(
        # 消息内容
        ('Hello python client, this is a async msg-%d' % i).encode('utf-8'),
        # 异步回调
        callback=send_callback,
        # 消息配置
        properties={'k': 'v'},
        # 业务key
        partition_key='yourKey'
    )

time.sleep(5)
client.close()