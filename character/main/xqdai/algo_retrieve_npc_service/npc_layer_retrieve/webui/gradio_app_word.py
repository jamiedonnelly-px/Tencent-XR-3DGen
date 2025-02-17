import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)

import argparse
import gradio as gr
import torch

import cv2
import numpy as np
import rembg
import torch
from PIL import Image

import uuid
import time
import json
import requests
import rpyc
import ujson

from ipdb import set_trace as st

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def deserialize(obj):
    if isinstance(obj, list):
        try:
            return np.array(obj)
        except ValueError:
            return [deserialize(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: deserialize(v) for k, v in obj.items()}
    else:
        return obj

def retrieve_text(prompt,gender,part):
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config["sync_request_timeout"] = None
    connection = rpyc.connect('', 0,config=rpyc_config)
    input = {'text_prompt':prompt,'gender':gender,'part':part}
    input_str = convert_to_serializable(input)
    input_json = ujson.dumps(input_str)
    output = connection.root.retrive_text(input_json)
    output = ujson.loads(output)
    output = deserialize(output)
    print(output)
    return output[0],output[1],output[2],output[3],output[4],output[5]

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
    custom_css = """
    body {
        background-color: lightblue;
    }

    .label {
        color: red;
        font-size: 20px;
    }
    """

    with gr.Blocks(
        title="NPC Retrieve - Web Demo",
        css=custom_css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
            # NPC Retrieve Demo
            """
            gr.Markdown(header)
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row(equal_height=True):
                    txt_input = gr.Textbox(
                        value='',
                        placeholder="please input text prompt",
                        show_label=False,
                        interactive=True)
            with gr.Column(scale=2):
                txt_btn = gr.Button(value="Retrieve using Text", variant="primary")
        with gr.Row(equal_height=True):
            dropdown_options_gender = ["male", "female"]
            gender_select = gr.Dropdown(choices=dropdown_options_gender,value="male",  label="Select gender",interactive=True)
        with gr.Row(equal_height=True):
            dropdown_options_parts = ["hair", "top","trousers","shoe","outfit","other"]
            part_select = gr.Dropdown(choices=dropdown_options_parts,value="hair",  label="Select part",interactive=True)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh1 = gr.Model3D(value=None, label="Mesh1", interactive=False)
                with gr.Row():
                    save_mesh2 = gr.Model3D(value=None, label="Mesh2", interactive=False)
               
            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh3 = gr.Model3D(value=None, label="Mesh3", interactive=False)
                with gr.Row():
                    save_mesh4 = gr.Model3D(value=None, label="Mesh4", interactive=False)

            with gr.Column(scale=1):
                with gr.Row():
                    save_mesh5 = gr.Model3D(value=None, label="Mesh5", interactive=False)
                with gr.Row():
                    save_mesh6 = gr.Model3D(value=None, label="Mesh6", interactive=False)
        
        txt_btn.click(
            fn=retrieve_text,
            inputs=[
                txt_input,gender_select,part_select
            ],
            outputs=[
                save_mesh1,save_mesh2,save_mesh3,
                save_mesh4,save_mesh5,save_mesh6,
            ]
        )

    launch_args = {"server_port": port}
    
    if listen:
        launch_args["server_name"] = "0.0.0.0"
        launch_args["inbrowser"] = True
    demo.queue().launch(**launch_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--port", type=int, default=8084)
    args = parser.parse_args()

    launch(
        args.port,
        listen=args.listen
    )


   
