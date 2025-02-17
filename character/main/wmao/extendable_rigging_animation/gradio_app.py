import os

import sys
sys.path.append(os.path.abspath(os.getcwd()))
import argparse
import gradio as gr
from ipdb import set_trace as st
import shutil
import subprocess
import numpy as np
from os import listdir
from omegaconf import OmegaConf
import glob
from easydict import EasyDict as edict

from main_fit_v2 import Fitter

CURRENT_PATH = os.getcwd()
OUT_DIR = f'{CURRENT_PATH}/output/web/'

args = {
    "config": f'{CURRENT_PATH}/config/fitting.yaml',
    "is_debug": True,
    "is_fitting": True,
    "is_rigging": True,
    "use_auto_weights": True
}

args = edict(args)
fitter = Fitter(args)

def save_file(ref_file):
    # st()
    file_name = os.path.basename(ref_file).split('.')[0]
    os.makedirs(f'{OUT_DIR}/{file_name}',exist_ok=True)
    out_file = f'{OUT_DIR}/{file_name}/{os.path.basename(ref_file)}'
    shutil.copyfile(ref_file, out_file)
    if ref_file.endswith('.obj'):
        directory = os.path.dirname(ref_file)
        for fn in os.listdir(directory):
            if os.path.isdir(f'{directory}/{fn}'):
                shutil.copytree(f'{directory}/{fn}', f"{OUT_DIR}/{file_name}/{fn}")
            elif os.path.isfile(f'{directory}/{fn}'):
                shutil.copyfile(f'{directory}/{fn}', f"{OUT_DIR}/{file_name}/{fn}")
    fitter.out_folder = f"{OUT_DIR}/{file_name}"
    fitter.update_target_mesh(out_file)
    return f'{fitter.out_folder}/fitting/target_mesh_vis.obj'

def auto_rig(ref_file, animal_type):
    fitter.reset_pose()
    fitter.fit()
    command = [
        "/root/blender-4.0.1-linux-x64/blender",
        "-b",
        "--python",
        "./blender/rigging.py",
        "--",
        "--template_fbx_dir", f"{CURRENT_PATH}/dataset/{animal_type}/template.fbx",
        "--rigging_info_dir", f"{fitter.out_folder}/rigging/rigging.npz",
        "--out_dir", f"{fitter.out_folder}/rigging/"
        ]
    print(' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    return f"{fitter.out_folder}/rigging/target_cano_final_vis.obj", f"{fitter.out_folder}/rigging/mesh_rigged.fbx"

def animate(ref_file, animal_type, motion_type):

    file_name = os.path.basename(ref_file).split('.')[0]
    command = [
        "/root/blender-3.5.0-linux-x64/blender",
        "-b",
        "--python",
        "./blender/animation.py",
        "--",
        "--src_animation", f"{CURRENT_PATH}/dataset/{animal_type}/animation/{motion_type}.FBX",
        "--tgt_fbx", f"{fitter.out_folder}/rigging/mesh_rigged.fbx",
        "--out_fbx", f"{fitter.out_folder}/animation/{motion_type}.fbx"
        ]
    print(' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return f"{fitter.out_folder}/animation/{motion_type}.fbx"

def animal_change(animal_type):
    motion_list = sorted(list(glob.glob(f"{CURRENT_PATH}/dataset/" + animal_type + '/animation/*.fbx')))  + sorted(list(glob.glob(f"{CURRENT_PATH}/dataset/"  + animal_type + '/animation/*.FBX')))
    motion_list_update = [os.path.basename(m).split('.')[0] for m in motion_list]
    fitter.update_template(f"{CURRENT_PATH}/dataset/" + animal_type)
    return gr.Dropdown(motion_list_update, label="Motion Type", info="Motion Type")

def animal_all(ref_file, animal_type):
    if animal_type == "Cat":
        prefix = 'tiger'
    elif animal_type == "Dog":
        prefix = 'canie'
    elif animal_type == "Cow":
        prefix = 'cattle'
    else: 
        prefix = 'cattle'
    motion_list = sorted(list(glob.glob(f"{CURRENT_PATH}/smal_motion_seq/{prefix}*")))
    motion_list_update = [os.path.basename(m).split('.')[0] for m in motion_list]
    for motion_type in motion_list_update:
        animate(ref_file, motion_type)

def launch(
    port,
    listen=False,
):
    lock_path = 'runing'

    global listen_port
    listen_port = port

    css = """
    #config-accordion, #logs-accordion {color: black !important;}
    .dark #config-accordion, .dark #logs-accordion {color: white !important;}
    .stop {background: darkred !important;}
    """
    
    animal_type = sorted(list(glob.glob(f"{CURRENT_PATH}/dataset/*/")))
    animal_type_list = [os.path.basename(os.path.normpath(p)) for p in animal_type]
    # get animations
    motion_list = sorted(list(glob.glob(animal_type[0]+'/animation/*.fbx'))) + sorted(list(glob.glob(animal_type[0]+'/animation/*.FBX')))
    motion_list_update = [os.path.basename(m).split('.')[0] for m in motion_list]
    fitter.update_template(f"{CURRENT_PATH}/dataset/" + animal_type_list[0])
    
    with gr.Blocks(
        title="Animal Auto Rigging Demo",
        css=css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
            Animal Auto Rigging Demo
            """
            gr.Markdown(header)
            
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Row():
                    ref_file = gr.File(
                        value=None,
                        label="Input mesh",
                        interactive=True
                        )
                with gr.Row():
                    ref_mesh = gr.Model3D(value=None, label="Uploaded Mesh", interactive=False)
                with gr.Row():
                    animal_type = gr.Radio(animal_type_list, value=animal_type_list[0],label="Animal Type")
                with gr.Row():
                    motion_type = gr.Dropdown(motion_list_update, label="Motion Type", info="Motion Type")
            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        rigging_btn = gr.Button(value="Auto Rigging", variant="primary")
                    with gr.Column(scale=1):
                        ani_btn = gr.Button(value="Generate Animation", variant="primary")
                with gr.Row():
                    can_mesh = gr.Model3D(value=None, label="Canonical Mesh", interactive=False)
                with gr.Row():
                    out_fbx = gr.File(value=None, label="output fbx",file_types=['filepath'], interactive=False)
                # with gr.Row():
                #     generate_all_btn = gr.Button(value="Generate All Animation", variant="primary")
        
        ref_file.upload(
            fn=save_file,
            inputs=[
                ref_file
            ],
            outputs=[
                ref_mesh
            ],
            concurrency_limit=1
        )

        rigging_btn.click(
            fn=auto_rig,
            inputs=[
                ref_file,
                animal_type,
            ],
            outputs=[
                can_mesh,
                out_fbx
            ],
            api_name="auto_rig",
            concurrency_limit=1
            
        )

        ani_btn.click(
            fn=animate,
            inputs=[
                ref_file,
                animal_type,
                motion_type,
            ],
            outputs=[
                out_fbx
            ],
            concurrency_limit=1
        )

        animal_type.change(
            fn=animal_change,
            inputs=[
                animal_type
            ],
            outputs=[
                motion_type
            ],
            concurrency_limit=1
        )

        # generate_all_btn.click(
        #     fn=animal_all,
        #     inputs=[
        #         ref_file,
        #         animal_type,
        #     ],
        #     outputs=[
        #     ],
        #     concurrency_limit=1
        # )

        # .success(
        #     fn=render,
        #     inputs=[
        #         ref_file,
        #         motion_type,
        #     ],
        #     outputs=[
        #         out_video
        #     ],
        #     concurrency_limit=1
        # )

    launch_args = {"server_port": port}
    if listen:
        launch_args["server_name"] = "0.0.0.0"
    demo.queue().launch(**launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()

    launch(
        args.port,
        listen=args.listen
    )