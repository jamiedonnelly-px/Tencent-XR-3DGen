import os
import shutil

def softlink_tmp_gradio(local_dir, shared_tmp_gradio_dir):
    if os.path.islink(local_dir):
        if local_dir == os.path.abspath(shared_tmp_gradio_dir):
            print(f'{local_dir} to {shared_tmp_gradio_dir} have linked. skip')
            return
        os.unlink(local_dir)
        print(f'unlink {local_dir} ')
        
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        if os.listdir(local_dir): 
            shutil.rmtree(local_dir)
            print(f"Removed directory {local_dir}")
    if os.path.exists(local_dir):
        if os.path.isdir(local_dir):
            os.rmdir(local_dir)
        else:
            os.remove(local_dir)
            
    # if os.path.exists(shared_tmp_gradio_dir):
    #     if os.path.islink(shared_tmp_gradio_dir):
    #         os.unlink(shared_tmp_gradio_dir)
    #     elif os.path.isdir(shared_tmp_gradio_dir):
    #         os.rmdir(shared_tmp_gradio_dir)
    #     else:
    #         os.remove(shared_tmp_gradio_dir)

    os.symlink(shared_tmp_gradio_dir, local_dir)

    print(f"Created symlink from {local_dir} to {shared_tmp_gradio_dir}")

local_dir = "/tmp/gradio"
shared_tmp_gradio_dir = "/aigc_cfs_3/sz/server/tmp_gradio"    
softlink_tmp_gradio(local_dir, shared_tmp_gradio_dir)