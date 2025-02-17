import os, pdb
import cv2
import numpy as np
import argparse
from PIL import Image

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--oname", "-o", required=True,
    help="This directory should include experiment specifications in 'specs_test.json'.",
)

args = arg_parser.parse_args()

basedir = os.path.join("output", args.oname)
path1 = os.path.join(basedir, "imgs100_-3")
path2 = os.path.join(basedir, "imgs100_1")
path3 = os.path.join(basedir, "imgs100_-2")

def path2pngs(path):
    pngs = os.listdir(path)
    pngs = sorted([x for x in pngs if x.endswith(".png")])
    pngs = [cv2.imread(os.path.join(path, x)) for x in pngs]
    return pngs

pngs1 = path2pngs(path1)
pngs2 = path2pngs(path2)
pngs3 = path2pngs(path3)

# pdb.set_trace()

num_frames = len(pngs1)
save_dir = os.path.join("output", args.oname, f"{args.oname}_tri")
os.makedirs(save_dir, exist_ok=True)
for i in range(num_frames):
    tri = np.concatenate([pngs1[i], pngs2[i], pngs3[i]], 1)[:,:,[2,1,0]]
    Image.fromarray(tri).save(os.path.join(save_dir, f"{i:05d}.jpg"))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fourcc = cv2.VideoWriter_fourcc(*'x264')

# out_video = cv2.VideoWriter(f'output/{args.oname}/{args.oname}_tri.mp4', fourcc, 25, (pngs1[0].shape[1]*3, pngs1[0].shape[0]))
# for i in range(num_frames):
#     out_video.write(np.concatenate([pngs1[i], pngs2[i], pngs3[i]], 1))
# out_video.release()