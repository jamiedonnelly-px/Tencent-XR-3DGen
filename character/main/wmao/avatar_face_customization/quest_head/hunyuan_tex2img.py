import requests
import json
import re
import random
import httpx
from openai import OpenAI

import os, sys
from pathlib import Path
ROOT_DIR = str(Path(os.getcwd()).parent)
sys.path.append(ROOT_DIR)
# from gRPC.pandorax_pb2 import TalkStreamReply

from ipdb import set_trace as st

client = OpenAI(
    api_key=None,
    base_url='http://url'
)

def ask_with_history(history):
    return client.images.generate(
        prompt="一张迪丽热巴的照片",
        model='hunyuan-image',
    ).data[0].url
    
def ask_with_history_stream(history):
    stream = client.chat.completions.create(
        messages=history,
        model='hunyuan',
        stream=True,
    )

    for chunk in stream:
        data = chunk.choices[0]
        ret = TalkStreamReply(
            role=data.delta.role,
            content=data.delta.content,
            finish_reason=data.finish_reason,
            )
        print(ret.content, end='')
        yield ret
        
def embeddings(input):
    embedding = client.embeddings.create(
        input=input,
        model='hunyuan-embedding',
    )
    # st()
    print(embedding.data[0].embedding)
    
def completion(prompt):
    completion = client.completions.create(
        model='hunyuan',
        prompt=prompt
    )
    st()

if __name__ == "__main__":
    # completion('this is a test')
    
    # history = [{"role": "user","content": "一张迪丽热巴的照片",}]
    history = [{"role": "user","prompt": "一张迪丽热巴的照片",}]
    
    ret = ask_with_history(history)
    st()
    print('done')
    # ask_with_history_stream(history)
    
    # embeddings("this is a test")