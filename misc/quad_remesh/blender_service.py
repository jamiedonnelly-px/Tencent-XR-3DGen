import sys, os
codedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(codedir)

from quad_converter import quad_remesh_pipeline
import argparse
import json
from glob import glob
import time
from pdb import set_trace as st
import traceback

def run_service(request_file):
    
    try:
        with open(request_file) as fi:
            request = json.load(fi)
        os.remove(request_file)
        job_id = request['job_id']
    except:
        return
    
    try:
        quad_remesh_pipeline(**request)
        success = True
        print(f"[INFO] successfully processed job {job_id}")
        feedback = "okay"
    except Exception as e:
        success = False
        feedback = "[Blender Error]: " + ''.join(traceback.format_exception(None, e, e.__traceback__))
    finally:
        os.makedirs(os.path.dirname(request["return_path"]), exist_ok=True)
        with open(request["return_path"], "w+") as fo:
            json.dump(dict(success=success, feedback=feedback), fo)

if __name__ == "__main__":
    
    # /path/to/blender -P blender_service.py -- -l "tmp" -c
    
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]
    
    print(argv)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen_dir", "-l", type=str, help="directory to listen to")
    parser.add_argument("--clear", "-c", action='store_true', help="clear listen dir before running")
    args = parser.parse_args(raw_argv)
    listen_dir = args.listen_dir
    
    print(f"starting blender process, listening on {os.path.abspath(listen_dir)}")
    
    if args.clear:
        for f in glob(os.path.join(listen_dir, '*.txt')):
            print(f"removing {f}")
            os.remove(f)
    
    while True:
        request_files = glob(os.path.join(listen_dir, '*.txt'))
        if len(request_files) > 0:
            ts = time.time()
            run_service(request_files[0])
            use_time = time.time() - ts
            print(f"[Log-Quad] use time = {use_time}")
        else:
            time.sleep(0.1)
        
    
    