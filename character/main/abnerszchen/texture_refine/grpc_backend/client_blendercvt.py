import os
import argparse
import logging
import grpc
import blendercvt_pb2
import blendercvt_pb2_grpc
from concurrent import futures


class BlenderCVTClient():
    def __init__(self, ip_port="localhost:987"):
        """client

        Args:
            client_cfg_json: abs path or relative name in codedir/configs. Defaults to 'client_texgen.json'.
            model_name: key of cfg, can be selected in webui
            device
        """
        self.ip_port = ip_port
        self.channel = grpc.insecure_channel(self.ip_port) 
        self.stub = blendercvt_pb2_grpc.BlendercvtStub(self.channel)
        logging.info(f"init blender cvt client to {self.ip_port} done")

    def interface_glb_to_obj(self, input_glb_path, output_obj_path):
        """convert glb to obj

        Args:
            input_glb_path: _description_
            output_obj_path: _description_

        Returns:
            True or False            
        """
        response = self.stub.ConvertGlbToObj(
            blendercvt_pb2.ConvertGlbToObjRequest(
                input_file=input_glb_path, output_file=output_obj_path
            )
        )
        print("[BlenderCVTClient] interface_glb_to_obj result:", response.message)
        return response.message == "True" and os.path.exists(output_obj_path)
      

    def interface_obj_to_glb(self, input_obj_path, output_glb_path):
        """convert obj to glb

        Args:
            input_obj_path: _description_
            output_glb_path: _description_

        Returns:
            True or False            
        """
        response = self.stub.ConvertObjToGlb(
            blendercvt_pb2.ConvertObjToGlbRequest(
                input_file=input_obj_path, output_file=output_glb_path
            )
        )
        print("[BlenderCVTClient] interface_obj_to_glb result:", response.message)
        return response.message == "True" and os.path.exists(output_glb_path)

    def interface_shutdown(self):
        """shutdown blender_worker. In general, it does not need to be called

        Args:
            ip_port: localhost:987 or ip:port

        Returns:
            True or False
        """
        response = self.stub.Shutdown(blendercvt_pb2.Empty())
        print("shutdown result:", response.message)
        return response.message


def test_client(lj_ip="localhost", lj_port="987"):
    ip_port = f"{lj_ip}:{lj_port}"
    client = BlenderCVTClient(ip_port)
    
    input_glb_path = "/aigc_cfs_2/sz/proj/tex_cq/dataset/uv_dataset/debug/glbs_old/out_000.glb"
    output_obj_path = "./debug/blender_cvt_out/mesh.obj"
    flag = client.interface_glb_to_obj(input_glb_path, output_obj_path)
    assert os.path.exists(output_obj_path), f"interface_glb_to_obj to {output_obj_path} failed"
    print('flag glb->obj', flag)
    
    input_obj_path = "/aigc_cfs_2/sz/proj/tex_cq/dataset/uv_dataset/debug/re-objs_old/out_000/mesh.obj"
    output_glb_path = "./debug/blender_cvt_out/mesh_re.glb"
    flag = client.interface_obj_to_glb(input_obj_path, output_glb_path)
    assert os.path.exists(output_glb_path), f"interface_obj_to_glb to {output_glb_path} failed"
    print('flag obj->glb', flag)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test blender cvt grpc')
    parser.add_argument('--lj_ip', type=str, default='localhost')
    parser.add_argument('--lj_port', type=str, default='987')
    args = parser.parse_args()

    test_client(args.lj_ip, args.lj_port)


# def interface_glb_to_obj(ip_port, input_glb_path, output_obj_path):
#     with grpc.insecure_channel(ip_port) as channel:
#         stub = blendercvt_pb2_grpc.BlendercvtStub(channel)
#         response = stub.ConvertGlbToObj(
#             blendercvt_pb2.ConvertGlbToObjRequest(
#                 input_file=input_glb_path, output_file=output_obj_path
#             )
#         )
#         print("interface_glb_to_obj result:", response.message)

# input_glb_path = "/aigc_cfs_2/sz/proj/tex_cq/dataset/uv_dataset/debug/glbs_old/out_000.glb"
# output_obj_path = "./debug/blender_cvt_out/mesh.obj"
# ip_port="localhost:987"
# interface_glb_to_obj(ip_port, input_glb_path, output_obj_path)
# assert os.path.exists(output_obj_path), f"interface_glb_to_obj to {output_obj_path} failed"
