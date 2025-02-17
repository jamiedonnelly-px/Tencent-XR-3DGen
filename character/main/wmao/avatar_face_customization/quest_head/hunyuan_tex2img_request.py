import requests
import json
import re
import random
from ipdb import set_trace as st

url = "http://url"
token = None
headers = {
    'Authorization': f'Bearer {token}'
}

def get_image(prompt='', negative_prompt='眼镜，帽子，头饰，发饰', style='古风二次元风格', out_dir=''):
    # data = {
    #     'model': 'hunyuan-image',
    #     'prompt': prompt,
    #     'negative_prompt': negative_prompt,
    #     'version': 'v1.8',
    #     'style': style
    # }
    data = {
        'model': 'hunyuan-image',
        'prompt': prompt+'，3D模型，光头，正脸，不戴眼镜，不戴帽子，不戴头饰，不戴发饰',
        'version': 'v1.8'
    }
    response = requests.post(url, headers=headers, json=data)
    ret = response.text
    ret = json.loads(ret)
    img_url = ret['data'][0]['url']
    save_path = f"{out_dir}/hunyuan_image.jpg"
    try:
        # Send a GET request to the URL
        response = requests.get(img_url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        # Write the content to a file in binary mode
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Image downloaded successfully and saved as {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
    return save_path

if __name__ == "__main__":

    prompt = '卡通男孩，3D模型，正脸，不戴眼镜，不戴帽子，不戴头饰，不戴发饰'
    negative_prompt = '眼镜，帽子，头饰，发饰'
    data = {
        'model': 'hunyuan-image',
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'version': 'v1.9',
        'style': '古风二次元风格'
    }
    response = requests.post(url, headers=headers, json=data)
    ret = response.text
    
    ret = json.loads(ret)
    img_url = ret['data'][0]['url']
    save_path = f"../output/{prompt}.jpg"
    print(img_url)
    try:
        # Send a GET request to the URL
        response = requests.get(img_url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        # Write the content to a file in binary mode
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Image downloaded successfully and saved as {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")