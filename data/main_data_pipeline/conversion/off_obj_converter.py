import argparse
import time

import trimesh

if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Verification starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(
        description='Remove mesh with certain material name.')
    parser.add_argument('--input_mesh_path', type=str, default="",
                        help='input off mesh file path')
    parser.add_argument('--output_mesh_path', type=str, default="",
                        help='output obj mesh file path')

    args = parser.parse_args()

    input_mesh_path = args.input_mesh_path
    output_mesh_path = args.output_mesh_path

    print("Transform off file at %s to %s" % (input_mesh_path, output_mesh_path))
    original_mesh = trimesh.load(input_mesh_path)
    original_mesh.export(output_mesh_path)
