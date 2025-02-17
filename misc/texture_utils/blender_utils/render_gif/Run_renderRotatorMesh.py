import sys
import platform
import os
import uuid
import subprocess
import importlib
import zipfile
import tarfile
from PIL import Image, ImageSequence
import imageio
import numpy as np
import time

fbx_path = r"E:\\blender\\render\\mesh3\\2a20b05f-3377-4de1-a65f-8241e918dbaf.glb"
out_put_dir = r"E:\\blender\\render\\out\\"
blender_file = r"E:\\blender\\Texture\\test1\\render_gif\\blender_render_gif\\RenderFactory_new_Black.blend"
python_file = r".\\RenderRotaterMesh.py"

def images_to_gif(images, output_path, duration=0.1, loop=0):
    # Convert images to Pillow format
    pil_images = [Image.fromarray(image) for image in images]

    # Save as GIF using Pillow
    pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], duration=duration*1000, loop=loop, optimize=True)


def create_gif(image_dir, gif_name, fps):
    frames = []

    files = os.listdir(image_dir)
    kwargs_write = {'fps':5.0, 'quantizer':'nq'}

    # 创建一个包含 (创建时间, 文件名) 的列表
    items_with_ctime = [(os.path.getctime(os.path.join(image_dir, item)), item) for item in files]

    # 根据创建时间进行排序
    sorted_items = sorted(items_with_ctime, key=lambda x: x[0])

    for ctime, file in sorted_items:
        #print(f"Item: {file}, Created Time: {ctime}")
        if os.path.basename(file).split(".")[1] in ['png', 'PNG']:
            filename = os.path.join(image_dir, file)
            frames.append(imageio.v3.imread(filename))
    
    imageio.mimsave(gif_name, frames, 'GIF', fps=fps, loop=0, palettesize=1024)
    #imageio.mimsave(gif_name, frames, 'GIF', **kwargs_write)

def read_images(images, image_dir):
    files = os.listdir(image_dir)

    # 创建一个包含 (创建时间, 文件名) 的列表
    items_with_ctime = [(os.path.getctime(os.path.join(image_dir, item)), item) for item in files]

    # 根据创建时间进行排序
    sorted_items = sorted(items_with_ctime, key=lambda x: x[0])

    for ctime, file in sorted_items:
        print(f"Item: {file}, Created Time: {ctime}")
        if os.path.basename(file).split(".")[1] in ['png', 'PNG']:
            filename = os.path.join(image_dir, file)
            images.append(imageio.v3.imread(filename))

if __name__ == '__main__':
    fbx_path = str(sys.argv[1])

    start_time = time.time()

    #根据文件名称输出渲染路径
    file_name = os.path.splitext(fbx_path)[0]
    out_put_dir = os.path.join(out_put_dir, file_name)
    if not os.path.isdir(out_put_dir):
        os.makedirs(out_put_dir)

    subprocess.run(['D:\\blender\\blender-3.6.11-candidate+v36.40cf7180d387-windows.amd64-release\\blender.exe', blender_file, '-b', '-P', python_file, '--',fbx_path, out_put_dir])

    tmp_dir = os.path.join(out_put_dir, "tmp")
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    gif_name = file_name + ".gif"
    gif_file = os.path.join(out_put_dir, gif_name)

    # 示例图像数据
    #images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(72)]
    images = []
    # read_images(images, tmp_dir)

    # 生成GIF
    # images_to_gif(images, gif_file)

    create_gif(tmp_dir, gif_file, 24)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")