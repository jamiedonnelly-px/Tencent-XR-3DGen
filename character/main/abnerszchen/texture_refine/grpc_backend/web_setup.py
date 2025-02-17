import gradio as gr
import os


def ui_setup(webclass, model_name_list):
    css = """
  #download {
    height: 50px;
  }
  """
    gr_path = os.path.dirname(__file__)
    with gr.Blocks(theme=gr.themes.Soft()) as block:
        # 1. row1 model in/out  and out image
        gr.HTML("<h1> Tex3D 请注意mcwy/lowpoly模型要使用对应的几何（可以用Example) </h1>")
        with gr.Row():
            gr.Markdown(value="请注意mcwy/lowpoly模型要使用对应的几何（可以用Example).\n 如果效果不满意请多尝试几次～ 可以拖动下方滑块调整运行参数")
        with gr.Row(variant='panel'):
            # input in left
            # with gr.Column(scale=1):
            ## input 3d model, image, text
            with gr.Column(scale=3):
                blk_model_in = gr.Model3D(
                    value=os.path.join(gr_path, "files/mcwy/mesh.glb"),
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="in model",
                )
            with gr.Column(scale=3):
                blk_model_out = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0], label="out model"
                )

            with gr.Column(scale=1):
                blk_image_out = gr.Image(type="filepath", label="out texture")

        # 2. row2 input image and text and button
        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                blk_image_in = gr.Image(
                    type="filepath",
                    value=os.path.join(gr_path, "files/6.png"),
                    label="condition image",
                    image_mode='RGBA',
                    # height=280,
                    # width=280,
                )
            with gr.Column(scale=1):
                blk_prompt = gr.Textbox(
                    label="Prompt",
                    elem_id=f"prompt",
                    show_label=False,
                    lines=6,
                    placeholder="Prompt",
                )
                blk_negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    elem_id=f"neg_prompt",
                    show_label=False,
                    lines=4,
                    placeholder="Negative prompt",
                )

            # button_like = gr.Button("like", icon="files/heart_empty_icon.ico")

            with gr.Column(scale=1):
                ## botton
                button_submit_text = gr.Button("Generate with Text only", variant='primary')
                button_submit_image = gr.Button("Generate with Image only", variant='primary')
                button_submit_mix = gr.Button("Generate with Text + Image", variant='primary')

                # gr.Markdown("need select model")
                button_reset_configs = gr.Button("reset default config", elem_id=f"button", variant="secondary")
                checkbox_debug = gr.Checkbox(True, label='debug mode(save temp logs..)')

        ## run configs
        with gr.Row(variant="compact"):
            with gr.Column(scale=1):
                # select model and cfg
                with gr.Row():
                    dropdown_model_list = gr.Dropdown(
                        choices=model_name_list,
                        label="style_models",
                        elem_id="dropdown",
                        value=model_name_list[0],
                    )

            with gr.Column(scale=1):

                ## input config
                slider_guidance_scale = gr.Slider(
                    5, 13, value=9, step=0.1, label="guidance_scale"
                )
                slider_num_inference_steps = gr.Slider(
                    10, 50, value=20, step=1, label="num_inference_steps"
                )     
            with gr.Column(scale=1):

                slider_controlnet_conditioning_scale = gr.Slider(
                    0, 1, value=0.7, step=0.1, label="Strength of control"
                )
                slider_ip_adapter_scale = gr.Slider(
                    0, 1, value=0.8, step=0.1, label="The weight of the image when mixing modes"
                )


        ## example
        with gr.Row(variant="compact"):
            gr.Examples(
                examples=[
                    [
                        os.path.join(gr_path, "files/mcwy/mcwy.glb"),
                        os.path.join(gr_path, "files/6.png"),
                    ],
                    [
                        os.path.join(gr_path, "files/mcwy/mcwy_top1.glb"),
                        os.path.join(gr_path, "files/mcwy/top1.png"),
                    ],
                    [
                        os.path.join(gr_path, "files/mcwy/mcwy_bottom1.glb"),
                        os.path.join(gr_path, "files/mcwy/bottom1.png"),
                    ],
                    # [
                    #     os.path.join(gr_path, "files/ready/top/ready.glb"),
                    #     os.path.join(gr_path, "files/ready/top/123.png"),
                    # ],                    
                    # [
                    #     os.path.join(gr_path, "files/lowpoly/lowpoly.glb"),
                    #     os.path.join(gr_path, "files/lowpoly/chip.png"),
                    # ],
                ],
                inputs=[blk_model_in, blk_image_in],
                cache_examples=False,
            )

        # 2. set func
        dropdown_model_list.change(
            webclass.select_models,
            inputs=[dropdown_model_list],
            outputs=[dropdown_model_list],
        )
        button_reset_configs.click(
            fn=webclass.reset_configs,
            outputs=[dropdown_model_list],
            # status_tracker=True,
        )

        button_submit_text.click(
            fn=webclass.pipe_text,
            inputs=[
                blk_model_in,
                blk_image_in,
                blk_prompt,
                blk_negative_prompt,
                slider_num_inference_steps,
                slider_guidance_scale,
                slider_controlnet_conditioning_scale,
                slider_ip_adapter_scale,
            ],
            outputs=[blk_model_out, blk_image_out],
        )
        button_submit_image.click(
            fn=webclass.pipe_image,
            inputs=[
                blk_model_in,
                blk_image_in,
                blk_prompt,
                blk_negative_prompt,
                slider_num_inference_steps,
                slider_guidance_scale,
                slider_controlnet_conditioning_scale,
                slider_ip_adapter_scale,
            ],
            outputs=[blk_model_out, blk_image_out],
        )        
        button_submit_mix.click(
            fn=webclass.pipe_mix,
            inputs=[
                blk_model_in,
                blk_image_in,
                blk_prompt,
                blk_negative_prompt,
                slider_num_inference_steps,
                slider_guidance_scale,
                slider_controlnet_conditioning_scale,
                slider_ip_adapter_scale,
            ],
            outputs=[blk_model_out, blk_image_out],
        )

        blk_prompt.value = "indian style"
        # blk_negative_prompt.value = "nsfw,paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), (grayscale), skin spots, acnes, skin blemishes,age spot,glan"
    return block
