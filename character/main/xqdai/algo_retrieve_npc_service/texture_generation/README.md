## 设置配置文件
配置文件:`base_interface/texture_generation/configs/client_texgen.json`   
一般只需要改`uv_mcwy`对应的数据, 如果用grpc接口需要设置文件里的`server_addr`, 如果用tdmq pulsar则不需要设置  

缓存路径可以设置`log_root_dir`, 其他运行参数可以不设置  

## grpc接口
`grpc_interface` 文件夹  

[grpc接口](grpc_interface/README.md)

## tdmq pulsar接口
`texture_generation` 文件夹  

[pulsar接口](tdmq_interface/README.md) 