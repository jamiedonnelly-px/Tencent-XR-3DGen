import os, pdb
import cv2

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--basedir", "-b", required=True,
    help="This directory should include experiment specifications in 'specs_test.json'.",
)

args = arg_parser.parse_args()

basedir = args.basedir
assert basedir[-1] == '/'
sname = basedir.split("/")[-2]
pngs = os.listdir(basedir)
pngs = sorted([x for x in pngs if x.endswith(".png")])
pngs = [cv2.imread(os.path.join(basedir, x)) for x in pngs]

# pdb.set_trace()

num_frames = len(pngs)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   
out_video = cv2.VideoWriter(f'{basedir}/../{sname}.mp4', fourcc, 25, (pngs[0].shape[1], pngs[0].shape[0]))
for i in range(num_frames):
    out_video.write(pngs[i])
out_video.release()