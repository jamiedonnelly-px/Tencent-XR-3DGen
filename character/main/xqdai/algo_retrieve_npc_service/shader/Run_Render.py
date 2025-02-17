import os
import subprocess
import cv2
import sys
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)


def RenderMesh(mesh_path):
    blenderFile_path = os.path.join(current_directory,"RenderFactory.blend")
    subprocess.run(['/root/blender-3.6.15-linux-x64/blender', blenderFile_path, '-b', '-P', os.path.join(current_directory,'FBX_RenderImage.py'),'--', mesh_path])

def render2video(mesh_path):
    out_image_path =os.path.join(os.path.dirname(mesh_path),'render')
    video_path = os.path.join(out_image_path, "render.webm")
    for index in range(6):
        name = 'Collection '+str(index+1)
        name = name.replace(' 1','')
        name = name.replace(' 2',' 2.001')
        filepath = os.path.join(out_image_path, str(name) + ".png")
        img = cv2.imread(filepath)
        if index==0:
            height, width, layers = img.shape
            fourcc = cv2.VideoWriter_fourcc(*'VP80') 
            video_writer = cv2.VideoWriter(video_path, fourcc, 2, (width, height))
        video_writer.write(img)
    video_writer.release()


if __name__ == "__main__":

    start_time = time.time()
    file_paths = [
        '/mnt/aigc_cfs_cq/xiaqiangdai/project/objaverse_retrieve/data/generate/f7811796-dab1-5273-9ff3-714493db8140/mesh/mesh.glb'
    ]   
    RenderMesh(file_paths[0])
    render2video(file_paths[0])
    end_time = time.time()
    print(f"cost time: {end_time - start_time} s")