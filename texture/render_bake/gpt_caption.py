import json
import base64

# pip install venus-api-base from tencent
from venus_api_base.http_client import HttpClient
from venus_api_base.config import Config
import json

def _gpt_caption(image_path):
    secret_id = None
    secret_key = None

    client = HttpClient(config=Config(read_timeout=15), secret_id=secret_id, secret_key=secret_key)

    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    header = {'Content-Type': 'application/json',}
    history = [
        {
            "role": "system",
            "content": "You are provided a grid image that contains either single or multiple views of the same single object, \
                    concisely describe this object (no the image) in english, \
                    try to capture as many specifics as possible WITHIN 50 WORDS. \
                    use plain sentences and reply directly with description itself and nothing else. start the sentence like \"a what kind of object of what features, style and overall appearance\""
        },
        {"role": "user","content": [
            {
            "type": "image_url",
            "image_url": {
                "detail": "high",
                "url": "data:image/png;base64,{}".format(encoded_string)
                        }
                    }
          ]}
    ]

    body = {
    "appGroupId": 3460,
    "model": "gpt-4-vision-preview",
    "messages": history,
    "temperature": 0.3,
    "max_tokens": 1000,
    }

    ret = client.post('http://venus-url', header=header, body=json.dumps(body))
    # print(ret['data']['response'])
    return ret['data']['response']

def gpt_caption(image_path, retries=4, default=""):
    for _ in range(retries):
        
        try:
            return _gpt_caption(image_path)
        except:
            continue
    
    return default

if __name__ == "__main__":

    
    # caption_string = gpt_caption('/aigc_cfs/Asset/designcenter/clothes/convert/readyplayerme/render/render_data/top/55/render_512_Valour/color/cam-0028.png')
    # print(caption_string)
    caption_string = gpt_caption("/aigc_cfs_2/zacheng/mvdream_control/depth-2-rgb/test_data/init/infer_image.png")
    print(caption_string)
    