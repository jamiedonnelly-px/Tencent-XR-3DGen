import os

import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.insert(0, parent_dir)

sys.path.append("/mnt/aigc_cfs_cq/xiaqiangdai/project/segment_anything_webui")
from inference import run_inference
import argparse
import gradio as gr
import torch

import cv2
import clip
import numpy as np
import rembg
import torch
from PIL import Image

import uuid
import time
import json
import requests

import sys
# sys.path.append('/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/bodyfit')
# from smpl_weights.run_auto_rig import auto_rig_layer

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

data_type = "readmeplay" #muchangwuyu,readmeplay,meishuzhongxin
readmeplay_PATH = '/mnt/aigc_cfs_gz/layer_avatar_data/readplayerMe/man_sep'
muchangwuyu_PATH = '/mnt/business_1/Data/DesignCenter/clock_fix_sample/20231215/fix_top_bottom/component'
meishuzhongxin_PATH = '/mnt/aigc_bucket_1/Asset/clothes/designcenter_part2/clothes'

FAISS_GPU_IDX_PATH = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_faiss_gpu_index_normalised.npy"
KEYS_PATH = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_keys.pkl'
EMB_2_OBJ = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/retrieve_libs/emb/NPC_emb_idx_2_obj_idx.pkl'
FAISS_GPU_IDX_PATH = FAISS_GPU_IDX_PATH.replace('/emb','/emb_'+data_type)
KEYS_PATH = KEYS_PATH.replace('/emb','/emb_'+data_type)
EMB_2_OBJ = EMB_2_OBJ.replace('/emb','/emb_'+data_type)
if data_type=="readmeplay":
    Global_PATH = readmeplay_PATH
elif data_type=="muchangwuyu":
    Global_PATH = muchangwuyu_PATH
else:
    Global_PATH = meishuzhongxin_PATH


TMP_PNG = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/test.png'

from CLIP_img_gen.retrieval import (
    load_faiss_index_gpu,
    load_emb_2_index,
    load_obj_keys,
    prepare_img,
    calc_img_emb,
    calc_txt_emb,
    retrive_topk_paths,
)

from ipdb import set_trace as st
    
def update_progress(progress, progress_path):
    with open(progress_path) as fi:
        prog = fi.read()
        # print(prog)
        try:
            title, ratio = prog.strip().split(' ')
            title = title.replace('_', ' ')
            ratio = float(ratio)
            progress(ratio, desc=title)
        except:
            print(f'can not parse progress: {prog}')

print(FAISS_GPU_IDX_PATH)
print(EMB_2_OBJ)
print(KEYS_PATH)
faiss_gpu_index = load_faiss_index_gpu(path = FAISS_GPU_IDX_PATH)
emb2obj = load_emb_2_index(path = EMB_2_OBJ)
obj_keys = load_obj_keys(path = KEYS_PATH)

# sdxl_pipe = load_SDXL_pipeline()
rembg_session = rembg.new_session()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14')
model.to(device)


def wrap_cloth( mesh_output_path, paths_temp ):
    obj_lst = os.path.join(mesh_output_path, "object_lst.txt")
    with open(obj_lst, "w") as f:
        for line in paths_temp:
            f.write(f"{line}\n")


    cmd = ["/root/miniconda3/envs/auto_rig/bin/python",
           "/mnt/aigc_cfs_cq/rabbityli/bodyfit/webui/cloth_warpper.py",
           "--lst_path",
           obj_lst]

    cmd = " ".join(cmd)
    os.system(cmd)
    return mesh_output_path

        
def txt_img(
    prompt,
    negative_prompt,
):
    print(prompt)

    images = sdxl_pipe(prompt=prompt, negative_prompt=negative_prompt,output_type="pil").images

    images[0].save(TMP_PNG)
    images = None
    
    torch.cuda.empty_cache()
    yield TMP_PNG

global g_paths
global g_paths_add
g_paths=[]
g_paths_add=[None,None,None,None]
def retrive(
    ref_image,
    progress=gr.Progress()
    ):
    
    progress(0, desc='removing background')
    
    # img = cv2.imread(ref_image, cv2.IMREAD_UNCHANGED)
    # img = cv2.resize(ref_image, (512, 512), interpolation=cv2.INTER_AREA)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(TMP_PNG, ref_image)
    
    img = Image.open(TMP_PNG).convert('RGBA')
    # img = rembg.remove(tmp, session=rembg_session)
    
    img = prepare_img(np.array(img))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    
    image_embed_norm = calc_img_emb(img, model, preprocess, device)
    paths = retrive_topk_paths(
        faiss_gpu_index, image_embed_norm,
        emb2obj, obj_keys,
        Global_PATH,
        10,
        data_type)
    
    if len(paths) < 3:
        for i in range(3-len(paths)):
            paths.append(None)
    global g_paths
    g_paths = paths[:3]
    print(paths[:3])
    return paths[:3]

def retrive_txt(prompt):
    query_txt_embed = calc_txt_emb(prompt, model, device)
    paths = retrive_topk_paths(
        faiss_gpu_index, query_txt_embed,
        emb2obj, obj_keys,
        Global_PATH,
        10,
        data_type)
    
    if len(paths) < 3:
        for i in range(3-len(paths)):
            paths.append(None)
    global g_paths
    g_paths = paths[:3]
    print(paths[:3])
    return paths[:3]

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
        title="NPC Retrive - Web Demo",
        css=custom_css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
            # NPC Retrive Demo
            """
            gr.Markdown(header)
                
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=2):
                with gr.Row().style(equal_height=True):
                    txt_input = gr.Textbox(
                        value='',
                        placeholder="Prompt",
                        show_label=False,
                        interactive=True)
            with gr.Column(scale=1):
                txt_btn = gr.Button(value="Retrive using Text", variant="primary")
        
        # SAM parameters
        with gr.Accordion(label='Parameters', open=False):
            with gr.Row():
                points_per_side = gr.Number(value=32, label="points_per_side", precision=0,
                                            info='''The number of points to be sampled along one side of the image. The total 
                                            number of points is points_per_side**2.''')
                pred_iou_thresh = gr.Slider(value=0.88, minimum=0, maximum=1.0, step=0.01, label="pred_iou_thresh",
                                            info='''A filtering threshold in [0,1], using the model's predicted mask quality.''')
                stability_score_thresh = gr.Slider(value=0.95, minimum=0, maximum=1.0, step=0.01, label="stability_score_thresh",
                                                info='''A filtering threshold in [0,1], using the stability of the mask under 
                                                changes to the cutoff used to binarize the model's mask predictions.''')
                min_mask_region_area = gr.Number(value=0, label="min_mask_region_area", precision=0,
                                                info='''If >0, postprocessing will be applied to remove disconnected regions 
                                                and holes in masks with area smaller than min_mask_region_area.''')
            with gr.Row():
                stability_score_offset = gr.Number(value=1, label="stability_score_offset",
                                                info='''The amount to shift the cutoff when calculated the stability score.''')
                box_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="box_nms_thresh",
                                        info='''The box IoU cutoff used by non-maximal ression to filter duplicate masks.''')
                crop_n_layers = gr.Number(value=0, label="crop_n_layers", precision=0,
                                        info='''If >0, mask prediction will be run again on crops of the image. 
                                        Sets the number of layers to run, where each layer has 2**i_layer number of image crops.''')
                crop_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="crop_nms_thresh",
                                            info='''The box IoU cutoff used by non-maximal suppression to filter duplicate 
                                            masks between different crops.''')

        with gr.Tab(label='Image'):
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    # input image
                    original_image = gr.State(value=None)   # store original image without points, default None
                    input_image = gr.Image(type="numpy")
                    # point prompt
                    with gr.Column():
                        selected_points = gr.State([])      # store points
                        with gr.Row():
                            gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                            undo_button = gr.Button('Undo point')
                        radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                     # text prompt to generate box prompt
                    text = gr.Textbox(label='Text prompt(optional)', info=
                    'If you type words, the OWL-ViT model will be used to detect the objects in the image, '
                    'and the boxes will be feed into SAM model to predict mask. Please use English.',
                                  placeholder='Multiple words are separated by commas')
                    owl_vit_threshold = gr.Slider(value=0.1, minimum=0, maximum=1.0, step=0.01, label="OWL ViT Object Detection threshold",
                                                info='''A small threshold will generate more objects, but may causing OOM. 
                                                A big threshold may not detect objects, resulting in an error ''')
                    # run button
                    button = gr.Button("Auto!")
                
                with gr.Column():
                    # show the image with mask
                    with gr.Tab(label='Image+Mask'):
                        output_image = gr.Image(type='numpy')
                    # show only mask
                    with gr.Tab(label='Mask'):
                        output_mask = gr.Image(type='numpy')
                
                    button_add_mask = gr.Button("add mask")

        def apply_mask(image, add_mask):
            if add_mask is None:
                return image
            gray_mask = cv2.cvtColor(add_mask, cv2.COLOR_BGR2GRAY)
    
            binary_mask = np.logical_and(gray_mask > 0, gray_mask < 255).astype(np.uint8)
            
            masked_image = image * np.stack([binary_mask]*3, axis=-1)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_x = float('inf')
            min_y = float('inf')
            max_x = 0
            max_y = 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

            width = max_x - min_x
            height = max_y - min_y
            x, y, w, h = min_x,min_y,width,height
            delta = 2
            x = np.max([0,x-delta])
            y = np.max([0,y-delta])
            w = np.min([w+2*delta,binary_mask.shape[1]])
            h = np.min([h+2*delta,binary_mask.shape[0]])
            cropped_image = masked_image[y:y+h, x:x+w]
 
            resized_image = cv2.resize(cropped_image, (512, 512))
            # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
            
            return resized_image

        
        with gr.Row().style(equal_height=False):
            with gr.Column(scale=1):
                with gr.Row():
                    image_output = gr.outputs.Image(label="Masked Image",type="numpy")
                button_add_mask.click(
                    apply_mask,
                    [original_image, output_mask],
                    [image_output])
                
                with gr.Row():
                    save_mesh1 = gr.Model3D(value=None, label="Save Mesh1", interactive=False)
                with gr.Row():
                    save_mesh2 = gr.Model3D(value=None, label="Save Mesh2", interactive=False)
                with gr.Row():
                    save_mesh3 = gr.Model3D(value=None, label="Save Mesh3", interactive=False)
                with gr.Row():
                    save_mesh4 = gr.Model3D(value=None, label="Save Mesh4", interactive=False)
            
            with gr.Column(scale=1):
                with gr.Row():
                    retrive_btn = gr.Button(value="Retrive using Image", variant="primary")
                with gr.Row():
                    output_mesh1 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                with gr.Column(scale=1):
                    output_mesh1_btn = gr.Button(value="select 1", variant="primary")
                with gr.Row():
                    output_mesh2 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                with gr.Column(scale=1):
                    output_mesh2_btn = gr.Button(value="select 2", variant="primary")
                with gr.Row():
                    output_mesh3 = gr.Model3D(value=None, label="Generated Mesh", interactive=False)
                with gr.Column(scale=1):
                    output_mesh3_btn = gr.Button(value="select 3", variant="primary")
                with gr.Column(scale=1):
                    clear_paths_btn = gr.Button(value="clear the last of saved models", variant="primary")
               
        
        with gr.Row().style(equal_height=False):
            with gr.Column(scale=1):
                with gr.Row():
                    model_generate_btn = gr.Button(value="model generation", variant="primary")
                with gr.Row():
                    with gr.Column(scale=1):
                        all_generate_mesh = gr.Model3D(value=None, label="all generate Mesh", interactive=False)
                    with gr.Column(scale=1):
                        animation_gif = gr.Image(type="numpy")

        # once user upload an image, the original image is stored in `original_image`
        def store_img(img):
            return img, []  # when new image is uploaded, `selected_points` should be empty
        input_image.upload(
            store_img,
            [input_image],
            [original_image, selected_points]
        )

        # user click the image to get points, and show the points on the image
        def get_point(img, sel_pix, point_type, evt: gr.SelectData):
            print("get_point")
            if point_type == 'foreground_point':
                sel_pix.append((evt.index, 1))   # append the foreground_point
            elif point_type == 'background_point':
                sel_pix.append((evt.index, 0))    # append the background_point
            else:
                sel_pix.append((evt.index, 1))    # default foreground_point
            # draw points
            print(sel_pix)
            for point, label in sel_pix:
                cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
            print(img.shape)
            # if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img if isinstance(img, np.ndarray) else np.array(img)
        input_image.select(
            get_point,
            [input_image, selected_points, radio],
            [input_image],
        )

        # undo the selected point
        def undo_points(orig_img, sel_pix):
            if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
                temp = cv2.imread(image_examples[orig_img][0])
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            else:
                temp = orig_img.copy()
            # draw points
            if len(sel_pix) != 0:
                sel_pix.pop()
                for point, label in sel_pix:
                    cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
            # if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            #     temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            return temp if isinstance(temp, np.ndarray) else np.array(temp)
        undo_button.click(
            undo_points,
            [original_image, selected_points],
            [input_image]
        )

        # button image
        # text =""
        # device ="cuda"
        # model_type ="vit_h"
        button.click(run_inference, inputs=[ points_per_side, pred_iou_thresh, stability_score_thresh,
                                        min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                        crop_nms_thresh, owl_vit_threshold, original_image, text, selected_points],
                    outputs=[output_image, output_mask])
        
        txt_btn.click(
            fn=retrive_txt,
            inputs=[
                txt_input
            ],
            outputs=[
                output_mesh1,
                output_mesh2,
                output_mesh3,
            ]
        )

        retrive_btn.click(
            fn=retrive,
            inputs=[
                image_output
            ],
            outputs=[
                output_mesh1,
                output_mesh2,
                output_mesh3,
            ]
        )
        global g_paths
        global g_paths_add
        output_sum = [save_mesh1,save_mesh2,save_mesh3,save_mesh4]
        def output_mesh1_select():
            if len(g_paths)>=1 and g_paths[0] not in g_paths_add:
                for i in range(4):
                    if g_paths_add[i]==None:
                        g_paths_add[i] = g_paths[0]
                        break
            print(g_paths_add)
            return g_paths_add

        output_mesh1_btn.click(
            fn=output_mesh1_select,
            outputs=output_sum
        )
        def output_mesh2_select():
            if len(g_paths)>=2 and g_paths[1] not in g_paths_add:
               for i in range(4):
                    if g_paths_add[i]==None:
                        g_paths_add[i] = g_paths[1]
                        break
            print(g_paths_add)
            return g_paths_add

        output_mesh2_btn.click(
            fn=output_mesh2_select,
            outputs=output_sum
        )
        def output_mesh3_select():
            if len(g_paths)>=3 and g_paths[2] not in g_paths_add:
               for i in range(4):
                    if g_paths_add[i]==None:
                        g_paths_add[i] = g_paths[2]
                        break
            print(g_paths_add)
            return g_paths_add
        output_mesh3_btn.click(
            fn=output_mesh3_select,
            outputs=output_sum
        )

        def clear_last_path():
            flag=0
            for i in range(len(g_paths_add)):
                if g_paths_add[i]==None and i>0:
                    g_paths_add.remove(g_paths_add[i-1])
                    g_paths_add.append(None)
                    flag=1
                elif g_paths_add[i]==None and i==0:
                    print("paths list None")
                    flag=2
            if flag==0:
                g_paths_add.remove(g_paths_add[len(g_paths_add)-1])
                g_paths_add.append(None)

            print(g_paths_add)
            return g_paths_add
              
        clear_paths_btn.click(
            fn=clear_last_path,
            outputs=output_sum
        )

    

        def model_generation():
            assert(None not in g_paths_add)
            timestamp = int(time.time())
            paths_temp = g_paths_add.copy()
        
            unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(timestamp))
            mesh_output_path = "/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/"+str(unique_id)
            if not os.path.exists(mesh_output_path):
                os.makedirs(mesh_output_path)

            if data_type == "muchangwuyu":
                paths_temp = [path.replace(".glb",".obj") for path in paths_temp]
            print(paths_temp)
            wrap_cloth( mesh_output_path, paths_temp)

            input = {"folder":mesh_output_path}
            json_data = json.dumps(input)
            headers = {"Content-Type": "application/json"}
            res = requests.post('', data=json_data,headers=headers)
            if res.status_code == 200:
                print("app_autoRig_layer combine Response:", res)
            else:
                print("app_autoRig_layer combine Request failed with status code")
                return None,None

            res = requests.post('', data=json_data,headers=headers)
            if res.status_code == 200:
                print("autoRig_layer auto_rig Response:", res)
            else:
                print("autoRig_layer auto_rig Request failed with status code")
                return None,None

            auto_rig_layer(mesh_output_path)

            print(mesh_output_path)
            mesh_output_path = '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/35e40bc3-3adb-55cc-872b-1c7f74a392f3'
            fbx_path = os.path.join(mesh_output_path,"mesh/mesh.fbx")
            gif_path = os.path.join(mesh_output_path,"mesh/mesh_animation.gif")
            input = {"fbx_path":fbx_path,
                    "gif_path":gif_path}
            json_data = json.dumps(input)
            headers = {"Content-Type": "application/json"}
            res = requests.post('', data=json_data,headers=headers)
            if res.status_code == 200:
                print("Response:", res)
            else:
                print("Request failed with status code")
                return None,None
            return os.path.join(mesh_output_path,"mesh/mesh.glb"),None
        
              
        model_generate_btn.click(
            fn=model_generation,
            outputs=[all_generate_mesh,animation_gif]
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
