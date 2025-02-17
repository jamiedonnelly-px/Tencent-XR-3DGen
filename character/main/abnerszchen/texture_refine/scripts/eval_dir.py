import os
import glob
import argparse

from common_utils import make_gif_dir


def eval_result_dir(in_dir, duration=1000):
    eval_out_dir = os.path.join(in_dir, 'eval')
    make_gif_dir(os.path.join(in_dir, 'train_vis'), os.path.join(eval_out_dir, 'train.gif'), duration=duration)
    make_gif_dir(os.path.join(in_dir, 'val_vis'), os.path.join(eval_out_dir, 'test.gif'), duration=duration)
    
    return

#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='render est obj list')
    parser.add_argument('in_dir', type=str)
    args = parser.parse_args()

    
    eval_result_dir(args.in_dir)
    return

if __name__ == "__main__":
    main()
