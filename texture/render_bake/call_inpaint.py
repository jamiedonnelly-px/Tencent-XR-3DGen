#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import random, os
import datetime

from pdb import set_trace as st

server_address = "ip_addr:80"
# client_id = str(uuid.uuid4())
client_id = str("inpaint_client")

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt, verbose=True):
    prompt_id = queue_prompt(prompt)['prompt_id']
    # print(prompt_id)
    output_images = {}
    while True:
        try:
            out = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if verbose:
            print(out)
        if isinstance(out, str):
            message = json.loads(out)
            # Finish by execution_success
            if message['type'] == 'execution_success':
                if message['data']["prompt_id"] == prompt_id:
                    if verbose:
                        print("Break:",message['data'].keys())
                    break #Execution is done

            # Finish by queue_remaining (Legacy support for old ComfyUI)
            # if message['type'] == 'status':
            #     data = message['data']
            #     if data['status']['exec_info']['queue_remaining']  == 0:
            #         if not ('sid' in message['data'].keys()):
            #             # print("Break")
            #             print(message['data'].keys())
            #             break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    # Save history as a json file with proper indent
    with open('history.json', 'w') as f:
        json.dump(history, f, indent=4)
    # st()
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
                output_images[node_id] = images_output

    return output_images


def inference_inpaint_flux(user_prompt, rgb_file, depth_file, mask_file, strength=0.75, seed=None, control_strength=0.6, width=1024, height=1024, circular_decode='disable', denoising_steps=12, verbose=True):
    """
        Cylindrical map inpainting
        Require:
            user_prompt # For general description of the object.
    """

    workflow_json_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow_api_cylindrical_inpaint.json')

    with open(workflow_json_txt) as json_file:
        prompt = json.load(json_file)

    # Set inputs
    # Input prompt
    prompt["6"]['inputs']['text'] = prompt["6"]['inputs']['text'].replace("user_prompt",user_prompt) # Concat text prompt to the original one
    # Output path
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    prompt["9"]['inputs']['filename_prefix'] = f"./inpaint_outputs/{current_date}/{current_time}_{user_prompt.replace(' ','_')[:100]}" # Subfolder under ComfyUI/outputs folder.
    # Seeds
    if seed is None:
        seed = random.randint(10**14, 10**15 - 1)
    prompt["25"]['inputs']['noise_seed'] = seed
    # Images
    prompt["65"]['inputs']['image'] = rgb_file
    prompt["57"]['inputs']['image'] = depth_file
    prompt["70"]['inputs']['image'] = mask_file
    # Denoising strength
    prompt["17"]['inputs']['denoise'] = strength
    # Depth control strength
    prompt["38"]['inputs']['strength'] = control_strength

    # Set resolution
    prompt["30"]['inputs']['width'] = width
    prompt["30"]['inputs']['height'] = height

    # Circular decode
    prompt["71"]['inputs']['tiling'] = circular_decode

    # Sampling steps
    prompt["17"]['inputs']['steps'] = denoising_steps

    # Call ComfyUI to generate images
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id), timeout=1)

    images = get_images(ws, prompt, verbose=verbose)

    # st()
    # print("***** images *****",images.keys())

    # Return generated image
    try:
        image_data = images['9'][0]
        from PIL import Image
        import io
        final_image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        print('[INPAINT] Error occurred in inpaint execution. No image generated.')
        final_image = None

    return final_image

def inference_inpaint_sdxl(user_prompt, rgb_file, depth_file, mask_file, strength=0.75, seed=None, control_strength=0.6, width=1024, height=1024, circular_decode='disable', denoising_steps=12, verbose=True):
    """
        Cylindrical map inpainting
        Require:
            user_prompt # For general description of the object.
    """

    workflow_json_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workflow_api_sdxl_inpaint_i2i_control_circular.json')

    with open(workflow_json_txt) as json_file:
        prompt = json.load(json_file)

    # Set inputs
    # Input prompt
    prompt["16"]['inputs']['text_g'] = prompt["16"]['inputs']['text_g'].replace("user_prompt",user_prompt) # Concat text prompt to the original one
    prompt["16"]['inputs']['text_l'] = prompt["16"]['inputs']['text_g']
    # Output path
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    prompt["9"]['inputs']['filename_prefix'] = f"./inpaint_outputs/{current_date}/{current_time}_sdxl_{user_prompt.replace(' ','_')[:100]}" # Subfolder under ComfyUI/outputs folder.
    # Seeds
    if seed is None:
        seed = random.randint(10**14, 10**15 - 1)
    prompt["36"]['inputs']['seed'] = seed
    # Images
    prompt["38"]['inputs']['image'] = rgb_file
    prompt["51"]['inputs']['image'] = depth_file
    prompt["47"]['inputs']['image'] = mask_file
    # Denoising strength
    prompt["36"]['inputs']['denoise'] = strength
    # Depth control strength
    prompt["50"]['inputs']['strength'] = control_strength

    # Set resolution
    prompt["16"]['inputs']['width'] = width
    prompt["16"]['inputs']['height'] = height
    prompt["16"]['inputs']['target_width'] = width
    prompt["16"]['inputs']['target_height'] = height
    prompt["19"]['inputs']['width'] = width
    prompt["19"]['inputs']['height'] = height
    prompt["19"]['inputs']['target_width'] = width
    prompt["19"]['inputs']['target_height'] = height

    # Circular decode
    prompt["54"]['inputs']['tiling'] = circular_decode

    # Sampling steps
    prompt["36"]['inputs']['steps'] = denoising_steps

    # Call ComfyUI to generate images
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id), timeout=1)

    images = get_images(ws, prompt, verbose=verbose)

    # st()
    # print("***** images *****",images.keys())

    # Return generated image
    try:
        image_data = images['9'][0]
        from PIL import Image
        import io
        final_image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        print('[INPAINT] Error occurred in inpaint execution. No image generated.')
        final_image = None

    return final_image

if __name__ == "__main__":

    inference_inpaint_flux(
        "Mario",
        rgb_file="cynlinder0.png",
        depth_file="depth0.png",
        mask_file="mask0.png"
    )


    # # Load workflow
    # workflow_json = '/aigc_cfs_2/jiayuuyang/ComfyUI/workflows/workflow_api_240814_cuteyou2.json'
    # with open(workflow_json) as f:
    #     prompt_text = f.read()

    # prompt = json.loads(prompt_text)

    # # Set inputs
    # # Input image 
    # prompt["912"]['inputs']['image'] = 'Pan-Ji-2.png' # A file in ComfyUI/input folder 
    # # Output path
    # current_date = datetime.datetime.now().strftime("%Y%m%d")
    # prompt["924"]['inputs']['filename_prefix'] = f'api_outputs/{current_date}_CuteYou2/out' # Subfolder under ComfyUI/outputs folder.
    # # Seed
    # prompt["354"]['inputs']['seed'] = random.randint(10**14, 10**15 - 1)
    # prompt["661"]['inputs']['seed'] = random.randint(10**14, 10**15 - 1)
    # # print("Seed: ", prompt["354"]['inputs']['seed'], prompt["661"]['inputs']['seed'])
    # # prompt["755"]['inputs']['seed'] = random.randint(10**14, 10**15 - 1)

    # # Send request
    # # queue_prompt(prompt) # Old version

    # ws = websocket.WebSocket()
    # ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

    # print("Waiting for execution to finish...")
    # images = get_images(ws, prompt)

    #Commented out code to display the output images:
    # ii = 0
    # for node_id in images:
    #     for image_data in images[node_id]:
    #         from PIL import Image
    #         import io
    #         image = Image.open(io.BytesIO(image_data))
    #         image.save(f"./api_outputs/test_{ii}.png")
    #         ii = ii + 1

    # Get final image
    # image_data = images['924']
    # from PIL import Image
    # import io
    # image = Image.open(io.BytesIO(image_data))
    # image.save(f"./api_outputs/test.png")
