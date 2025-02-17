import logging
import argparse
import os
import grpc
from concurrent import futures

import sys

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)
sys.path.append(os.path.join(codedir, "grpc_backend"))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


import blendercvt_pb2
import blendercvt_pb2_grpc
import multiprocessing

codedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(codedir)

from dataset.uv_dataset.glb_obj_converter import blender_worker


class BlenderCVTServer(blendercvt_pb2_grpc.BlendercvtServicer):
    def __init__(self):
        super(BlenderCVTServer, self).__init__()

        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()

        self.blender_process = multiprocessing.Process(
            target=blender_worker,
            args=(self.request_queue, self.response_queue),
        )
        self.blender_process.start()

    def ConvertGlbToObj(self, request, context):
        self.request_queue.put(
            {
                "type": "convert_glb_to_obj",
                "input_file": request.input_file,
                "output_file": request.output_file,
            }
        )
        result = self.response_queue.get()
        return blendercvt_pb2.ConvertReply(message=f"{result}")

    def ConvertObjToGlb(self, request, context):
        self.request_queue.put(
            {
                "type": "convert_obj_to_glb",
                "input_file": request.input_file,
                "output_file": request.output_file,
            }
        )
        result = self.response_queue.get()
        return blendercvt_pb2.ConvertReply(message=f"{result}")

    def Shutdown(self, request, context):
        self.request_queue.put("shutdown")
        self.blender_process.join()
        result = True
        return blendercvt_pb2.ConvertReply(message=f"{result}")


def main_server(lj_port="987"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    blendercvt_pb2_grpc.add_BlendercvtServicer_to_server(BlenderCVTServer(), server)
    server.add_insecure_port(f"[::]:{lj_port}")
    server.start()
    print(f"BlenderCVTServer started, listening on {lj_port}")
    server.wait_for_termination()


if __name__ == "__main__":
    """
    /usr/blender-3.6.2-linux-x64/blender -b -P server_blendercvt.py 
    """
    main_server("987")
