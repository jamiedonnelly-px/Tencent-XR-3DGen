import sys
import argparse
import os
import subprocess
import time

from PIL import Image
import imageio
import uuid
import numpy as np
# import imageio.v3 as imageio
import shutil
import traceback

def init_job_id():
    return str(uuid.uuid4())

code_dir = os.path.dirname(os.path.abspath(__file__))

def delete_folder_after_delay(folder_path, delay=200):
    command = [
        'nohup',  # 使用nohup命令来防止进程被挂起
        sys.executable,  # Python解释器
        '-c',
        (
            f'import time, shutil, os; '
            f'time.sleep({delay}); '
            f'folder_path="{folder_path}"; '
            f'os.path.exists(folder_path) and shutil.rmtree(folder_path)'
        ),
        '&'  # 在后台运行
    ]
    
    # 启动一个新的后台进程并立即返回
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
def images_to_gif(images, output_path, duration=0.1, loop=0):
    # Convert images to Pillow format
    pil_images = [Image.fromarray(image) for image in images]

    # Save as GIF using Pillow
    pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], duration=duration*1000, loop=loop, optimize=True)


def create_gif(image_dir, gif_name, fps):
    ts = time.time()
    frames = []

    files = os.listdir(image_dir)
    kwargs_write = {'fps':5.0, 'quantizer':'nq'}

    # 创建一个包含 (创建时间, 文件名) 的列表
    items_with_ctime = [(os.path.getctime(os.path.join(image_dir, item)), item) for item in files]

    # 根据创建时间进行排序
    sorted_items = sorted(items_with_ctime, key=lambda x: x[0])

    filenames = []
    for ctime, file in sorted_items:
        print(f"Item: {file}, Created Time: {ctime}")
        if os.path.basename(file).split(".")[1] in ['png', 'PNG']:
            filename = os.path.join(image_dir, file)
            filenames.append(filename)
         # frames.append(imageio.v3.imread(filename))
    t_load = time.time()
    
    frames = [imageio.v3.imread(filename) for filename in filenames]   
    
    print('load imgs done')
    # imageio.mimsave(gif_name, frames, 'GIF', fps=fps, loop=0, palettesize=1024)
    # imageio.mimsave(gif_name, frames, 'GIF', **kwargs_write)
    # imageio.mimsave(gif_name, frames, 'GIF', fps=fps)    # TODO
    imageio.mimsave(
        gif_name, frames, 'GIF', 
        fps=fps, 
        # palettesize=128,  # 调整调色板大小
        # subrectangles=True,  # 启用子矩形编码
        # quantizer='nq'  # 使用更高效的量化器
    )
        
    t_done = time.time()
    
    print(f"load imgs use time={t_load - ts}, gif use time {t_done - t_load}, all time = {t_done-ts}")

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


class RenderMeshGif:
    def __init__(self, blender_path="/usr/blender-3.6.2-linux-x64/blender",
                 blend_file="RenderFactory_new_Black.blend",
                 python_file="RenderRotaterMesh.py"):
        self.blender_path = blender_path
        self.blend_file = os.path.join(code_dir, blend_file)
        self.python_file = os.path.join(code_dir, python_file)

        for need_file in [self.blender_path, self.blend_file, self.python_file]:
            assert os.path.exists(need_file), need_file        

        self.temp_dir = "/blender_render_gif"
        # self.temp_dir = "/mount/blender_gif"
        os.makedirs(self.temp_dir, exist_ok=True)
        

    def call_render_gif(self, in_mesh, out_gif, render_in_fly=True):
        try:
            ts = time.time()
            assert os.path.exists(in_mesh), in_mesh
            if render_in_fly:
                out_put_dir_fly = os.path.join(self.temp_dir, init_job_id())
            else:
                out_put_dir_fly = os.path.join(os.path.dirname(out_gif), 'render_gif')
            
            os.makedirs(out_put_dir_fly, exist_ok=True)
            subprocess.run([self.blender_path, self.blend_file, '-b', '-P', self.python_file, '--', in_mesh, out_put_dir_fly])
            t_blender = time.time()


            tmp_dir = os.path.join(out_put_dir_fly, "tmp")
            if not os.path.isdir(tmp_dir):
                os.makedirs(tmp_dir)
            
            file_name = os.path.splitext(os.path.basename(in_mesh))[0]
            if render_in_fly:
                gif_file = os.path.join(out_put_dir_fly, f"{file_name}.gif")
            else:
                gif_file = out_gif

            # 示例图像数据
            #images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(72)]
            images = []
            # read_images(images, tmp_dir)

            # 生成GIF
            # images_to_gif(images, gif_file)

            # breakpoint()
            create_gif(tmp_dir, gif_file, 12)
            
            os.makedirs(os.path.dirname(out_gif), exist_ok=True)
            if render_in_fly:
                shutil.copyfile(gif_file, out_gif)
                delete_folder_after_delay(out_put_dir_fly)
            
            t_done = time.time()
            print(f"blender use time={t_blender - ts}, gif use time{t_done - t_blender}, all time = {t_done-ts}")
            return True
        except Exception as e:
            print(f"call_render_gif e={e}")
            traceback.print_exc()
            return False
            
if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser(description='blender render.')
    parser.add_argument('--in_mesh', type=str,
                        default="/aigc_cfs_gdp/sz/threeviews/e3514f8e-9c47-4ee3-a216-f5eb98a0cde7/texall/textured.glb",
                        help='path to source mesh')
    parser.add_argument('--out_gif', type=str,
                        default="/aigc_cfs_gdp/sz/threeviews/e3514f8e-9c47-4ee3-a216-f5eb98a0cde7/texall/768_32.gif",
                        help='fout_put_dir')
    parser.add_argument('--blender_path',
                        type=str,
                        default="/usr/blender-3.6.2-linux-x64/blender",
                        help='blender path')    
    parser.add_argument('--blend_file',
                        type=str,
                        default="RenderFactory_new_Black.blend",
                        help='blend file, relative path')
    parser.add_argument('--python_file',
                        type=str,
                        default="RenderRotaterMesh.py",
                        help='python path')    
    args = parser.parse_args()
    
    in_mesh = args.in_mesh
    out_gif = args.out_gif
    blender_path = args.blender_path
    blend_file = args.blend_file
    python_file = args.python_file

    render_mesh_gif = RenderMeshGif(blender_path, blend_file, python_file)
    render_mesh_gif.call_render_gif(in_mesh, out_gif)