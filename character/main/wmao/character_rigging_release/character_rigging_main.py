import numpy as np
import trimesh
from ipdb import set_trace as st
import trimesh.scene
import open3d as o3d
import argparse
import sys
import json
from joint_2d_to_3d import main as joint2d_to_3d
from joint_2d_to_3d import main_v2 as joint2d_to_3d_v2
from joint_2d_to_3d import main_v3 as joint2d_to_3d_v3
from joint_2d_to_3d import main_v4 as joint2d_to_3d_v4
import subprocess
import os
import time
from datetime import datetime
from gputool.generate_gif import images_to_gif
from gputool.tdmq_client_gputool import GpuToolInterface, init_job_id
import shutil

def main(mesh_path='',joint_web='', out_dir='', 
        animation_type="Walking", is_animation=True, is_rigging=True, 
        is_standardize=False, blender_gif_interface=None,
        rig_hand=True,
        model=None,
        model_partseg=None,
        check_armature=False):
    """_summary_

    Args:
        mesh_path (str, optional): old: mesh path on gdp new: mesh path at cos.  Defaults to ''.
        joint_web (str, optional): 2d joints position from webui. Defaults to ''.
        out_dir (str, optional): old: parse from mesh path, new: parse from redis. Defaults to ''.
        animation_type (str, optional): must be one of ["Walking","Jumping","Running","Dancing","Boxing"]. Defaults to "Walking".
        is_animation (bool, optional): _description_. Defaults to True.
        is_rigging (bool, optional): _description_. Defaults to True.
        blender_gif_interface (_type_, optional): interface used to render gif. Defaults to None.
        rig_hand (bool, optional): _description_. Defaults to True.
        model (nn.model, optional): _description_. Defaults to True.
        model_partseg (nn.model, optional): _description_. Defaults to True.

    Returns:
        rigged_dir (str, optional): old: mesh path on gdp new: mesh path at cos.  Defaults to ''.
        animation_dir (str, optional): old: mesh path on gdp new: mesh path at cos.  Defaults to ''.
        mesh_path (str, optional): old: mesh path on gdp new: mesh path at cos.  Defaults to ''.
    """    """"""
    # Get the current date and time
    now = datetime.now()
    # Full date and time
    # formatted_full = now.strftime("%Y-%m-%d %H:%M:%S")
    # print("Task start at ", formatted_full)
    src_ani="/aigc_cfs_gdp/weimao/character_rigging/animations"
    ani_list = ["Walking","Jumping","Running","Dancing","Boxing"]

    if animation_type not in ani_list:
        animation_type = np.random.choice(ani_list)
    
    if animation_type == 'Jumping':
        animation_type = 'Jumping2'
        
    src_ani = src_ani + '/' + animation_type + '.fbx'
    
    if isinstance(mesh_path,list):
        mesh_path = mesh_path[0]
        
    # if not os.path.exists(mesh_path):
    #     mesh_path = '/mnt/aigc_bucket_4/pandorax/quad_remesh/' + mesh_path.split('/data/general_generate/')[1]
    #     print("new mesh path", mesh_path)

    if isinstance(joint_web, list):
        joint_web = joint_web[0]
    out_dir = out_dir + '/character_rigging'
    os.makedirs(out_dir, exist_ok=True)

    # copy original mesh to out_dir
    try:
        shutil.copy(mesh_path, out_dir + '/' + os.path.basename(mesh_path))
    except:
        print('copy file error')
    mesh_path = out_dir + '/' + os.path.basename(mesh_path)
    mesh_basename = os.path.splitext(os.path.basename(mesh_path))[0]

    output_gif = None

    # check if armature exists
    if check_armature:
        start_time = time.time()
        command = [
            "/usr/blender-4.0.1-linux-x64/blender",
            "-b",
            "--python",
            "./blender_script/blender_check_armature.py",
            "--",
            "--target_file", mesh_path,
            "--out_dir", out_dir
            ]
        print(' '.join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        has_armature = os.path.join(out_dir,'has_armature.npy')
        has_armature = np.load(has_armature).item()
        print(f'check armature time: {time.time() - start_time:.3f} second')
        return has_armature


    # standardize the mesh
    if is_standardize:
        start_time = time.time()
        command = [
            "/usr/blender-4.0.1-linux-x64/blender",
            "-b",
            "--python",
            "./blender_script/blender_standardize.py",
            "--",
            "--target_file", mesh_path,
            "--out_dir", out_dir
            ]
        print(' '.join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        mesh_standardized = os.path.join(out_dir, mesh_basename + "_standardized.glb")
        print(f'standardlize mesh time: {time.time() - start_time:.3f} second')
        return mesh_standardized

    rigged_dir = out_dir + '/mesh_rigged.fbx'
    if os.path.exists(rigged_dir.replace('.fbx','_new.fbx')):
        rigged_dir = rigged_dir.replace('.fbx','_new.fbx')
    ani_name = os.path.basename(src_ani)
    ani_name = ani_name.split('.')[0]
    animation_dir = out_dir + f'/mesh_{ani_name}.fbx'
    
    if is_rigging:
        start_time = time.time()
        if isinstance(joint_web, dict):
            joints2d_web = joint_web
        elif isinstance(joint_web, str):
            if os.path.exists(joint_web):
                with open(joint_web,'r') as f:
                    joints2d_web = json.load(f)
            else:
                joint_web = joint_web.replace("'",'"')
                joints2d_web = json.loads(joint_web)
        with open(out_dir + '/joint_web.json', "w") as f:
            json.dump(joints2d_web, f)
        RoateX = joints2d_web["RoateX"]
        RoateY = joints2d_web["RoateY"]
        RoateZ = joints2d_web["RoateZ"]
        
        # Scale = joints2d_web["Scale"]
        joints2d_web = joints2d_web["Points"]
        joints2d = {}
        for jweb in joints2d_web:
            joints2d[jweb['label']] = jweb['rigpoint']

        # preprocess mesh
        command = [
            "/usr/blender-4.0.1-linux-x64/blender",
            "-b",
            "--python",
            "./blender_script/blender_preprocess.py",
            "--",
            "--target_dir", mesh_path,
            "--out_file", out_dir + '/mesh_mainifold_rotated.obj',
            "--threejs_x", str(RoateX),
            "--threejs_y", str(RoateY),
            "--threejs_z", str(RoateZ)
            ]
        print(' '.join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f'manifold mesh time: {time.time() - start_time:.3f} second')

        # joint2d_to_3d(mesh_path, joints2d, out_dir + '/joint3d.npz', threejs_x=RoateX, threejs_y=RoateY, threejs_z=RoateZ)
        # joint2d_to_3d_v2(out_dir + '/mesh_mainifold_rotated.obj', joints2d, out_dir + '/joint3d_v2.npz', threejs_x=RoateX, threejs_y=RoateY, threejs_z=RoateZ)
        # joint2d_to_3d_v3(out_dir + '/mesh_mainifold_rotated.obj', joints2d, out_dir + '/joint3d_v2.npz', skeleton_type='openxr')
        joint2d_to_3d_v4(out_dir + '/mesh_mainifold_rotated.obj', out_dir + '/mesh_rotated.obj' , 
                        joints2d, out_dir + '/joint3d_v2.npz', skeleton_type='openxr',rig_hand=rig_hand,
                        model=model, model_partseg=model_partseg)
        command = [
            "/usr/blender-4.0.1-linux-x64/blender",
            "-b",
            "--python",
            "./blender_script/blender_rigging_new.py",
            "--",
            "--joint_dir", out_dir + '/joint3d_v2.npz',
            "--target_dir", mesh_path,
            "--template_fbx_dir", "/aigc_cfs_gdp/weimao/character_rigging/akai_e_espiritu.fbx",
            "--out_file", rigged_dir,
            "--threejs_x", str(RoateX),
            "--threejs_y", str(RoateY),
            "--threejs_z", str(RoateZ)
            ]
        print(' '.join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f'rigging time: {time.time() - start_time:.3f} second')
        
        
    if is_animation:
        # start_time = time.time()
        # command = [
        #     "/usr/blender-4.0.1-linux-x64/blender",
        #     "-b",
        #     "--python",
        #     "./blender_animation_new.py",
        #     "--",
        #     "--src_animation", src_ani,
        #     "--tgt_fbx", rigged_dir,
        #     "--out_fbx", animation_dir
        #     ]
        # print(' '.join(command))
        # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # stdout, stderr = process.communicate()
        # print(f'animation time: {time.time() - start_time:.3f} second')

        start_time = time.time()
        command = [
            "/usr/blender-4.0.1-linux-x64/blender",
            "-b",
            "--addons",
            "rokoko-studio-live-blender-master",
            "--python",
            "./blender_script/blender_animation_rokoko.py",
            "--",
            src_ani,
            rigged_dir.replace('.fbx','.glb') if is_rigging else mesh_path,
            animation_dir.replace('.fbx','.glb')
            ]
        print(' '.join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f'animation time: {time.time() - start_time:.3f} second')

        # # render gif
        # blender_path =  "/usr/blender-3.6.2-linux-x64/blender"
        # blend_file = "/aigc_cfs_gdp/weimao/character_rigging/gputool/RenderFactory_Black_remove_light.blend"
        # python_file = "/aigc_cfs_gdp/weimao/character_rigging/gputool/render_wrot.py"
        # in_mesh = animation_dir.replace('.fbx','_new.glb')
        # out_put_dir_fly = os.path.dirname(animation_dir)
        # subprocess.run([blender_path, blend_file, '-b', '-P', python_file, '--', in_mesh, out_put_dir_fly])
        # output_gif = animation_dir.replace('.fbx','.gif')
        # images_to_gif(out_put_dir_fly + '/tmp/', output_gif)
        
        if blender_gif_interface:
            start_time = time.time()
            query_job_ids = set()
            job_id = init_job_id()
            query_job_ids.add(job_id)
            in_mesh = animation_dir.replace('.fbx','.glb')
            output_gif = animation_dir.replace('.fbx','.gif')
            success, out_gif = blender_gif_interface.blocking_call_render_gif(job_id, in_mesh, output_gif)
            print(f'generate gif success {success}, used {time.time()-start_time:.3f}s')

    return rigged_dir.replace('.fbx', '.glb'), animation_dir.replace('.fbx','.glb'), output_gif


if __name__ == "__main__":
    
    try:
        from hand_rigging.models.point_transformer_partseg import PointTransformerSeg38
        from hand_rigging.get_hand_joint import remove_noise_points, compute_principal_directions
        from hand_rigging.utils import part_colors
        import torch
    except:
        print('cannot import hand rigging tool')
    

    in_channels = 3
    out_feat = 128
    n_parts = 6
    model = PointTransformerSeg38(in_channels=in_channels, num_classes=out_feat).float().cuda() 
    model_partseg = torch.nn.Linear(in_features=out_feat, out_features=n_parts).float().cuda()
    checkpoint = torch.load('./hand_rigging/ckpt/best_model.pth')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model_partseg.load_state_dict(checkpoint['model_partseg_state_dict'])
    model.eval()
    model_partseg.eval()
    print(f'load model at epoch {start_epoch}')
    
    jobid = '05355263-d87c-44db-9ae5-9d62c7ad446d'
    mesh_path = f'/aigc_cfs_gdp/jiawei/data/general_generate/{jobid}/character_rigging/05355263-d87c-44db-9ae5-9d62c7ad446d.glb'
    joint_web = f'/aigc_cfs_gdp/jiawei/data/general_generate/{jobid}/character_rigging/joint_web.json'
    out_dir = f'/aigc_cfs_gdp/jiawei/data/general_generate/{jobid}'
    # mesh_path = '/aigc_cfs_gdp/jiawei/data/character_rigging/7644096b-da70-4b46-883f-c1ace18497cd/character_rigging/mesh_mainifold_rotated.obj'
    # joint_web = '/aigc_cfs_gdp/jiawei/data/character_rigging/7644096b-da70-4b46-883f-c1ace18497cd/character_rigging/joint_web.json'

    # blender_gif_interface = GpuToolInterface("/aigc_cfs_gdp/weimao/character_rigging/gputool/configs/tdmq_gputool.json")
    blender_gif_interface = None
    a = main(mesh_path, joint_web, out_dir, 
            animation_type="Walking", is_standardize=False, blender_gif_interface=blender_gif_interface,
            model=model, model_partseg=model_partseg)
    # blender_gif_interface.close() 
    print(a)