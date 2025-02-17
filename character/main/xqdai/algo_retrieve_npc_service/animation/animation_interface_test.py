import os
import time
import random
import json
from distribute_logging import *
import datetime
import subprocess

current_file_path = os.path.abspath(__file__)
current_file_directory = os.path.dirname(current_file_path)
from motion_text_retrieve.retrieval_bge_embeding import mixamo_h3d_text_retrieve,mixamo_h3d_text_retrieve_multi_output
from motion_text_retrieve.retrieval_bge_embeding_mixamo import mixamo_text_retrieve
os.environ['blender']="/root/blender-3.6.15-linux-x64/blender"
if not os.path.exists("/root/blender-3.6.15-linux-x64"):
    print("cp -r /aigc_cfs_gdp/xiaqiangdai/envs//blender-3.6.15-linux-x64.tar.gz /root/")
    print("cd /root/ && tar -zxvf blender-3.6.15-linux-x64.tar.gz")
    os.system("rm /root/blender-* -rf")
    os.system("cp -r /aigc_cfs_gdp/xiaqiangdai/envs/blender-3.6.15-linux-x64.tar.gz /root/")
    os.system("cd /root/ && tar -zxvf blender-3.6.15-linux-x64.tar.gz")
    os.system("rm /root/blender-3.6.15-linux-x64.tar.gz")

mixamo_models_folder = '/aigc_cfs_gdp/xiaqiangdai/data/mixamo_models/xiaqiangdai'
mixed_models_folder = '/aigc_cfs_gdp/xiaqiangdai/data/HumanML3D/models'
mixed_models_gif_folder = '/aigc_cfs_gdp/xiaqiangdai/data/HumanML3D/models_gif'

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Output:", result.returncode)
        return 0
    except:
        return -1

def mdm_process():
    os.system("cd "+current_file_directory+"/motion-diffusion-model/ && /apdcephfs/private_neoshang/software/anaconda3/envs/mdm/bin/python -m sample.generate_v --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 1 --num_repetitions 1")

def animation_mdm_process(fbx_path,fbx_out_path,rotations_all_file,root_translation_file):
    cmd= "cd "+current_file_directory+" && "+os.environ.get('blender')+"  -b  -P  "+os.path.join(current_file_directory,'smpl_drive_mdm_extend.py')+" -- "+fbx_path+" "+fbx_out_path+' '+rotations_all_file+" "+root_translation_file
    return os.system(cmd)

def animation_transfer_process(character_path_src,character_path_dst,character_path_out):
    print(character_path_src)
    print(character_path_dst)
    cmd = os.environ.get('blender')+"  --background --addons rokoko-studio-live-blender-master --python  "+os.path.join(current_file_directory,'retarget_test.py')+" -- "+character_path_src+" "+character_path_dst+" "+character_path_out
    return run_command(cmd)

def merge_animation(character_path_in,character_path_idle,character_path_out):
    cmd = os.environ.get('blender')+"  --background  --python  "+os.path.join(current_file_directory,'merge_animation.py')+" -- "+character_path_in+" "+character_path_idle+" "+character_path_out
    return run_command(cmd)


def animation_render(fbx_path,gif_path):
    cmd= "cd "+current_file_directory+" && "+os.environ.get('blender')+"  -b -P  "+os.path.join(current_file_directory,'smpl_drive_mdm_render.py')+" -- "+fbx_path+" && python "+os.path.join(current_file_directory,'imgs2gif.py')+" -- "+fbx_path+" "+gif_path
    return os.system(cmd)

def get_random_file(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return os.path.join(path, random.choice(files)) if files else None

def animation(fbx_path, gif_path,logger=None):
    fbx_out_path = fbx_path.replace(".fbx","_animation.fbx").replace(".glb","_animation.glb")
    if logger==None:
        print(f"{fbx_path} {gif_path}")
    else:
        logger.info(f"{fbx_path} {gif_path}")
    text_save = ''
    prob = random.randint(0, 100)
    if prob<2:
        index = random.randint(0, 8)
        while index==1 or index==2:
            index = random.randint(0, 8)
        rotation_file = current_file_directory+"/motion-diffusion-model/drive_vector/drive_rotations_all_"+str(index)+".npy"
        root_file = current_file_directory+"/motion-diffusion-model/drive_vector/drive_root_translation_"+str(index)+".npy"

        start_time = time.time()
        ret = animation_mdm_process(fbx_path,fbx_out_path,rotation_file,root_file)
        if 0!=ret:
            raise ValueError(f"animation_process error {ret}")

        text_save = rotation_file+' '+root_file
        end_time = time.time()
        elapsed_time = end_time - start_time  
        print(f"animation_mdm_process Time: {elapsed_time:.2f} seconds")
    else:

        start_time = time.time()
        character_path_src = get_random_file(mixamo_models_folder)
        ret = animation_transfer_process(character_path_src.replace(' ','\ '),fbx_path.replace(' ','\ '))
        if 0!=ret:
            raise ValueError(f"animation_process error {ret}")
        
        text_save = character_path_src
        end_time = time.time()
        elapsed_time = end_time - start_time  
        print(f"animation_transfer_process Time: {elapsed_time:.2f} seconds")

    parent_dir = os.path.dirname(fbx_path)
    json_file_path = os.path.join(parent_dir, 'animation.json')
    with open(json_file_path, "w") as json_file:
        json.dump(text_save, json_file)

    return 0

def animation_text(fbx_path, text_prompt,logger=None):
    fbx_out_path = fbx_path.replace(".fbx","_animation.fbx").replace(".glb","_animation.glb")
    os.system('rm '+fbx_out_path)
    parent_dir = os.path.dirname(fbx_path)
    json_file_path = os.path.join(parent_dir, 'animation.json')

    start_time = time.time()
    if text_prompt!='':
        if os.path.exists(json_file_path):
            if logger==None:
                print(f"{fbx_path} {json_file_path} exists")
            else:
                logger.info(f"{fbx_path} {json_file_path} exists")
            with open(json_file_path, "r") as file:
                data = json.load(file)
                keys = list(data.keys())
                character_path_src = data[keys[0]]
                if keys[0]!=text_prompt:
                    retrieve_result = mixamo_h3d_text_retrieve(text_prompt)
                    if logger==None:
                        print(retrieve_result)
                    else:
                        logger.info(f"{fbx_path} {text_prompt} {retrieve_result}")

                    character_path_src = os.path.join(mixamo_models_folder,retrieve_result+'.fbx')
                    if not os.path.exists(character_path_src):
                        character_path_src = os.path.join(mixed_models_folder,retrieve_result+'.fbx')
        else:
            if logger==None:
                print(f"{fbx_path} {json_file_path} not exists")
            else:
                logger.info(f"{fbx_path} {json_file_path} not exists")
            retrieve_result = mixamo_h3d_text_retrieve(text_prompt)
            if logger==None:
                print(retrieve_result)
            else:
                logger.info(f"{fbx_path} {text_prompt} {retrieve_result}")
            
            character_path_src = os.path.join(mixamo_models_folder,retrieve_result+'.fbx')
            if not os.path.exists(character_path_src):
                character_path_src = os.path.join(mixed_models_folder,retrieve_result+'.fbx')

        output_filepath = fbx_path.replace('.fbx','_animation.fbx').replace('.glb','_animation.glb')
        ret = animation_transfer_process(character_path_src.replace(' ','\ '),fbx_path.replace(' ','\ '),output_filepath.replace(' ','\ '))
        if 0!=ret:
            raise ValueError(f"animation_process error {ret}")
    else:
        character_path_srcs = ['/aigc_cfs_gdp/xiaqiangdai/data/mixamo_idle/Idle.fbx','/aigc_cfs_gdp/xiaqiangdai/data/others/SK_AIGC_mesh_in.fbx','/aigc_cfs_gdp/xiaqiangdai/data/others/SK_AIGC_mesh_out.fbx']
        output_filepath_idle = fbx_path.replace('.fbx','_animation_idle.fbx').replace('.glb','_animation_idle.glb')
        ret = animation_transfer_process(character_path_srcs[0].replace(' ','\ '),fbx_path.replace(' ','\ '),output_filepath_idle.replace(' ','\ '))
        if 0!=ret:
            raise ValueError(f"animation_process error {ret}")
        output_filepath_in = fbx_path.replace('.fbx','_animation_in.fbx').replace('.glb','_animation_in.glb')
        ret = animation_transfer_process(character_path_srcs[1].replace(' ','\ '),fbx_path.replace(' ','\ '),output_filepath_in.replace(' ','\ '))
        if 0!=ret:
            raise ValueError(f"animation_process error {ret}")
        output_filepath_out = fbx_path.replace('.fbx','_animation_out.fbx').replace('.glb','_animation_out.glb')
        ret = animation_transfer_process(character_path_srcs[2].replace(' ','\ '),fbx_path.replace(' ','\ '),output_filepath_out.replace(' ','\ '))
        if 0!=ret:
            raise ValueError(f"animation_process error {ret}")
        
        ret = merge_animation(output_filepath_in,output_filepath_idle,output_filepath_out)
        if 0!=ret:
            raise ValueError(f"animation_process error {ret}")
        
        character_path_src = character_path_srcs[0]
    
    
    if not os.path.exists(fbx_out_path):
        os.system('rm /root/blender-3.6.15-linux-x64 -rf  && cp /aigc_cfs_gdp/xiaqiangdai/envs/blender-3.6.15-linux-x64.tar.gz /root/ && cd /root/ && tar -zxvf blender-3.6.15-linux-x64.tar.gz')
        if logger!=None:
            logger.error(f"{fbx_path} animation error")
        else:
            print((f"{fbx_path} animation error"))
        raise ValueError(f"animation_process error")
    
    text_save = character_path_src
    end_time = time.time()
    elapsed_time = end_time - start_time  
    if logger==None:
        print(f"animation_transfer_process Time: {elapsed_time:.2f} seconds")
    else:
        logger.info(f"{fbx_path} animation_transfer_process Time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    with open(json_file_path, "w") as json_file:
        save_dict = {text_prompt:text_save}
        json.dump(save_dict, json_file)
    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"animation json output Time: {elapsed_time:.2f} seconds")
    return 0

def animation_text_retrieve(text_prompt,logger=None):
    if text_prompt=='':
        raise ValueError(f"animation_text_retrieve text null")
    names_list = mixamo_h3d_text_retrieve_multi_output(text_prompt)
    output_gif_path = []
    for name in names_list:
        gif_path = os.path.join(mixamo_models_folder,name+'.gif')
        if not os.path.exists(gif_path):
            gif_path = os.path.join(mixed_models_gif_folder,name+'.gif')
            if not os.path.exists(gif_path):
                continue
        output_gif_path.append(gif_path)

    return output_gif_path

def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension

def animation_text_retrieve_retarget(gif_path,target_model_path,logger=None):
    if gif_path=='':
        print("animation_text_retrieve_retarget gif_path null")
        raise ValueError(f"animation_text_retrieve_retarget gif_path null")
    if not os.path.exists(gif_path):
        print("animation_text_retrieve_retarget gif_path not exist")
        raise ValueError(f"animation_text_retrieve_retarget gif_path not exist")
    if not os.path.exists(target_model_path):
        print("animation_text_retrieve_retarget target_model_path not exist")
        raise ValueError(f"animation_text_retrieve_retarget target_model_path not exist")
    
    src_model_path =  os.path.join(mixed_models_folder,gif_path.split('/')[-1].replace('.gif','.fbx'))
    if not os.path.exists(src_model_path):
        print("animation_text_retrieve_retarget target_model_path not exist")
        raise ValueError(f"animation_text_retrieve_retarget src_model_path not exist")
    start_time = time.time()
    # output_filepath = target_model_path.replace(".fbx","_animation.fbx").replace(".glb","_animation.glb")
    output_filepath = target_model_path.replace(".fbx","_animation.fbx").replace(".glb","_animation.glb").replace('/mnt/aigc_bucket_4/pandorax/retrieve_NPC','/aigc_cfs_gdp/xiaqiangdai/retrieveNPC_save')
    output_filepath = os.path.join(os.path.dirname(output_filepath),f"mesh_animation{get_file_extension(output_filepath)}")
    print(output_filepath)
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath),exist_ok=True)
    cmd = "cp "+target_model_path+' '+os.path.dirname(output_filepath)
    ret = os.system(cmd)
    if 0!=ret:
        print(f"animation_process cp {ret}")
        raise ValueError(f"animation_process cp {ret}")
    target_model_path_temp = os.path.join(os.path.dirname(output_filepath),target_model_path.split('/')[-1])
    print(target_model_path_temp,output_filepath)
    ret = animation_transfer_process(src_model_path.replace(' ','\ '),target_model_path_temp.replace(' ','\ '),output_filepath.replace(' ','\ '))
    if 0!=ret:
        print(f"animation_process error {ret}")
        raise ValueError(f"animation_process error {ret}")
    parent_dir = os.path.dirname(target_model_path)
    json_file_path = os.path.join(parent_dir, 'animation.json')

    text_save = src_model_path
    end_time = time.time()
    elapsed_time = end_time - start_time 
    if logger==None:
        print(f"animation_transfer_process Time: {elapsed_time:.2f} seconds")
    else:
        logger.info(f"{target_model_path} animation_transfer_process Time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    with open(json_file_path, "w") as json_file:
        save_dict = {gif_path:text_save}
        json.dump(save_dict, json_file)
    end_time = time.time()
    elapsed_time = end_time - start_time 
    print(f"animation json output Time: {elapsed_time:.2f} seconds")
    return output_filepath
  

if __name__ == '__main__':
    start_time = time.time()
    glb_path = '/mnt/aigc_bucket_4/pandorax/retrieve_NPC/retrieve_NPC/bb1c7ceb-7850-4ccd-b830-18f95b531169/character_rigging/mesh_Walking.glb'
    gif_path = '/mnt/aigc_bucket_4/pandorax/retrieve_NPC/d4c08fe0-e3f3-5bdc-a4c4-8d93df88295f/007086.gif'
    # animation_text(fbx_path,'')
    # print(animation_text_retrieve('run'))
    result = animation_text_retrieve_retarget(gif_path,glb_path)
    print(result)
    end_time = time.time()  

    elapsed_time = end_time - start_time  

    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    