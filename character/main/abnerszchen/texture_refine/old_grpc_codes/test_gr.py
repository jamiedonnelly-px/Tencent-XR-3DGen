import gradio as gr
import os
from easydict import EasyDict as edict

# test_dict = {"a":2}
# data = edict(test_dict)
# print("b" in data)

def load_mesh(mesh_file_name):
    return mesh_file_name

demo = gr.Interface(
    fn=load_mesh,
    inputs=gr.Model3D(),
    outputs=gr.Model3D(
            clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model"),
    cache_examples=True
)

if __name__ == "__main__":
    # demo.launch(server_port=80)
    demo.launch(server_port=80, server_name='0.0.0.0')

