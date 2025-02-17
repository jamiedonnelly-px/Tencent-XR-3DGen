
## 运行测试
如果没有安装grpc环境，可以运行以下虚拟环境：
```
source /aigc_cfs_2/sz/grpc/bin/activate
```
也可以自己pip安装, 依赖很少:
```
pip install -r ./requirements.txt
```


调用grpc服务: ip和配置预先在配置文件`configs/client_texgen.json`中设置好(默认不改)：
```
python client_texgen.py  configs/client_texgen.json --model_key uv_mcwy
```
model_key是在配置文件中的key，选模型  



如果需要重新生成proto：
```
bash scripts/gen_code.sh
```

## 接口
```
## init
client = TexGenClient(client_cfg_json, model_name=model_key)
```

#### 纯文本模式
```

job_id = init_job_id()
out_mesh_paths_query_key = client.webui_query_text(
    job_id,
    in_mesh_path,
    in_prompts,
    in_mesh_key="BR_TOP_1_F_T",
    out_objs_dir=os.path.join(out_objs_dir, "query_key"),
)
"""webui query text only mode

Args:
    job_id(string): uuid
    in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
    in_prompts(string/ list of string): 文本提示,可以是字符串或字符串list
    in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
    out_objs_dir(string): 输出文件夹路径
Returns:
    out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
"""
```

#### 纯图片模式
```
out_mesh_paths_image = client.webui_query_image(
    init_job_id(),
    in_mesh_path,
    in_condi_img,
    in_mesh_key=None,
    out_objs_dir=os.path.join(out_objs_dir, "image"),
)
"""webui query image only mode

Args:
    job_id(string): uuid
    in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
    in_condi_img(string): 控制图片路径. condi img path
    in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
    out_objs_dir(string): 输出文件夹路径
Returns:
    out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
"""
```

#### 文图混合模式

```
out_mesh_paths_mix = client.webui_query_text_image(
    init_job_id(),
    in_mesh_path,
    in_prompts,
    in_condi_img,
    in_mesh_key=None,
    out_objs_dir=None,
)
"""webui query text+image mode

Args:
    job_id(string): uuid
    in_mesh_path(string): mesh绝对路径. raw obj/glb path with uv coord
    in_prompts(string/ list of string): 文本提示,可以是字符串或字符串list
    in_condi_img(string): 控制图片路径. condi img path
    in_mesh_key(string): 类似BR_TOP_1_F_T这样的检索到的key, 如果有的话最好提供, 没有的话可以不给
    out_objs_dir(string): 输出文件夹路径 if None use default dir(cfg.log_root_dir)
Returns:
    out_mesh_paths: list of out_mesh_path or "", If "" is returned, the task failed.
"""
```

#### 通用接口
三个模式实际上是用一个统一函数封装的, 可以直接调统一函数, 不过不是很建议,需要的话使用:
```
client.client_obj_tex_gen()
```