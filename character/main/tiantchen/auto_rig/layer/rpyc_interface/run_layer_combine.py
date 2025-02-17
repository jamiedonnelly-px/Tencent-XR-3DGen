import os, time, logging
logging.basicConfig(filename='/aigc_cfs/tinatchen/auto_rig/layer/tdmq_interface/example1.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def auto_rig_layer(input_path):
    time1 = time.time()
    dirname, _ = os.path.split(os.path.abspath(__file__))
    print(dirname)
    cmd = " ".join(["/root/tinatchen/blender-3.6.13-linux-x64/blender -b -P {}".format(os.path.join(dirname, "../rpyc_interface/layer_combine.py")),"--", input_path])#, "> /dev/null"])
    os.system(cmd)
    time2 = time.time()
    print("total spent time: {:.2f}".format(time2-time1))
    print(input_path)

    job_id = os.path.basename(input_path)
    logging.info(f"rpyc run done job_id:{job_id}")
    logging.info(f"use time:{time2-time1}")

def auto_rig_layer_test(input_path):
    time1 = time.time()
    dirname, _ = os.path.split(os.path.abspath(__file__))
    print(dirname)
    cmd = " ".join(["/root/tinatchen/blender-3.6.13-linux-x64/blender -b -P {}".format(os.path.join(dirname, "../rpyc_interface/layer_combine_copy.py")),"--", input_path, "> /dev/null"])
    os.system(cmd)
    time2 = time.time()
    print("total spent time: {:.2f}".format(time2-time1))

def auto_rig_layer_docker(input_path):
    time1 = time.time()
    dirname, _ = os.path.split(os.path.abspath(__file__))
    print(dirname)
    cmd = " ".join(["/root/blender-3.6.5-linux-x64/blender -b -P {}".format("/aigc_cfs/tinatchen/auto_rig/layer/rpyc_interface/layer_combine.py"),"--", input_path])#, "> /dev/null"])
    os.system(cmd)
    time2 = time.time()
    print("total spent time: {:.2f}".format(time2-time1))

def layer_preprocess(path):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    cmd_1 = " ".join(["/root/tinatchen/blender-3.6.5-linux-x64/blender -b -P {}".format(os.path.join(dirname, "preprocess.py")),"--", path])
    os.system(cmd_1)

if __name__ == '__main__':
   
    in_mesh_path = "/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save/b141c276-9d1c-5ae2-a43e-0a458b12e28d"
    auto_rig_layer(in_mesh_path)

    # for i in []: 
        # "MCWY2_F_T", "MCWY2_M_T", "mcwy_female", "mcwy_male"
        # "readyplayerme_male", "readyplayerme_male_T"
        # "timer_female", "timer_male"
        # "pubg_female", "pubg_male"
        # "yuanmeng"
        # "quest_female_average", "quest_male_average", "quest_male_fat", "quest_male_slim", "quest_male_strong"
        # layer_preprocess(os.path.join("/aigc_cfs/rabbityli/base_bodies", i))
    

    # with open('/aigc_cfs/Asset/designcenter/clothes/convert/mcwy2/fix_A_pose/fbx_list.txt', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         dirname, _ = os.path.split(os.path.abspath(__file__))
    #         cmd_1 = " ".join(["/root/tinatchen/blender-3.6.5-linux-x64/blender -b -P {}".format(os.path.join(dirname, "../smpl_weights/skinning_weight/A_T.py")), "--", line.strip()])
    #         os.system(cmd_1)

