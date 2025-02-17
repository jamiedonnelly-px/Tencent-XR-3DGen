## 环境
```
pip install pulsar-client==3.5.0
pip install redis
```

[文档](https://doc.weixin.qq.com/doc/w3_AXgAuAbKAKIkhEMjFOWQjOhSvfF6E?scode=AJEAIQdfAAo6pqmzQfAXgAuAbKAKI)

## 极简式接口-类grpc逻辑
[默认config路径:../configs/client_texgen.json](../configs/client_texgen.json), 包含消息队列的token, url和topic；及redis信息

```
python easy_interface.py
```


## 纹理生成消费者
[默认config路径:../configs/tex_gen.json](../configs/tex_gen.json), 包含消息队列的token, url和topic
在每台机器上部署(后续GDP批量部署):
```
python consumer_texgen.py
```

配合blender服务化加速:
```
bash run_pulsar_texgen.sh
```


## 前后端接口
代码包含同步用法和异步用法  
[默认config路径:../configs/client_texgen.json](../configs/client_texgen.json), 包含消息队列的token, url和topic；及redis信息

```
python main_call_texgen.py
```
包含:  

### 1. 前端推送纹理生成请求
`TexGenProducer`, 接口包括文本模式\图片模式\文图混合模式  
topic=`pulsar-32j8nb4k7393/aigc-x-1/aigc-topic-tex-generation`
#### 纯文本模式
```
producer = TexGenProducer(args.client_cfg_json, args.model_name)
producer.interface_query_text(
    job_id,
    in_mesh_path,
    in_prompts,
    in_mesh_key=in_mesh_key,
    out_objs_dir=os.path.join(out_objs_dir, f"query_key_{i}"),
)
"""query text only mode

Args:
    job_id(string): uuid
    in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
    in_prompts(string/ list of string): 文本提示,可以是字符串或字符串list
    in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
    out_objs_dir(string): 输出文件夹路径
Returns:
    send_flag: send succeed or failed
"""
```
#### 纯图片模式
```
producer.interface_query_image(job_id,
                                in_mesh_path,
                                in_condi_img,
                                in_mesh_key=in_mesh_key,
                                out_objs_dir=os.path.join(out_objs_dir, f"query_image"))
"""query image only mode

Args:
    job_id(string): uuid
    in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
    in_condi_img(string): 控制图片路径. condi img path
    in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
    out_objs_dir(string): 输出文件夹路径
Returns:
    send_flag: send succeed or failed
"""
```

#### 文图混合模式
```
producer.interface_query_text_image(job_id,
                                    in_mesh_path,
                                    in_prompts,
                                    in_condi_img,
                                    in_mesh_key=in_mesh_key,
                                    out_objs_dir=os.path.join(out_objs_dir, f"query_mix"))
"""query text+image mode

Args:
    job_id(string): uuid
    in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
    in_prompts(string/ list of string): 文本提示,可以是字符串或字符串list
    in_condi_img(string): 控制图片路径. condi img path
    in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
    out_objs_dir(string): 输出文件夹路径
Returns:
    send_flag: send succeed or failed
"""
```

### 2. 后端接收纹理生产的结果
`BackendConsumer`  监听纹理替换的结果, topic=`pulsar-32j8nb4k7393/aigc-x-1/aigc-topic-backend-tex`  
接收的数据体格式和处理示例如下所示, 后端同学可以二次开发
```
def parse_and_run(self, msg):
    """receive result of texture_generation

    Args:
        msg: json.loads(msg.data()) = :
        out_dict = {
            "service_name": "texture_generation",
            "job_id": job_id,   # str uuid
            "flag": success,    # T/F
            "result": result,   # 输出的结果list of obj path or [""]
            "feedback": feedback # 报错信息(如果有)
        }
    Returns:
        job_id
        suc_flag=T/F
        results= list of obj path or [""]
    """
```