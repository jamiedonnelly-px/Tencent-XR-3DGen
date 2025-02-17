# from pymxs import runtime as mxs
import time

import pymxs as mxs


def to_valid_path(string: str):
    new_string = string.replace("\\", "\\\\")
    return new_string


if __name__ == '__main__':
    opts = mxs.runtime.maxops.mxsCmdLineArgs
    max_file_path = opts[mxs.runtime.Name("max_file_path")]
    output_file_path = opts[mxs.runtime.Name("output_file_path")]

    max_file_path = to_valid_path(max_file_path)
    output_file_path = to_valid_path(output_file_path)

    mxs.runtime.FBXExporterSetParam('Animation', False)
    mxs.runtime.FBXExporterSetParam('ASCII', False)
    mxs.runtime.FBXExporterSetParam('Cameras', False)
    mxs.runtime.loadMaxFile(max_file_path)
    time.sleep(0.1)
    mxs.runtime.exportFile(output_file_path, mxs.runtime.Name("noPrompt"), using='FBXEXP')
    mxs.runtime.resetMaxFile(mxs.runtime.Name("noPrompt"))
