import argparse
import os
import sys
import time

import bpy

if __name__ == '__main__':
    argv = sys.argv
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Install addons on blender on server')
    parser.add_argument('--addon_path', type=str,
                        help='path of addon zip file to be installed')
    args = parser.parse_args(raw_argv)

    addon_path = args.addon_path

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    addon_folder = os.path.split(addon_path)[0]
    addon_name = os.path.split(addon_path)[1]
    addon_basename = os.path.splitext(addon_name)[0]

    bpy.ops.preferences.addon_install(filepath=addon_path)
    bpy.ops.preferences.addon_enable(module=addon_basename)
    bpy.ops.wm.save_userpref()

    time.sleep(0.1)
