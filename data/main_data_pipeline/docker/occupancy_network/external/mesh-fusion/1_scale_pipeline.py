import os
import time
import common
import argparse
import numpy as np


def run_scale(input_mesh_path: str, output_mesh_path: str, output_transformation_txt: str, padding: float):
    mesh = common.Mesh.from_obj(input_mesh_path)

    # Get extents of model.
    bb_min, bb_max = mesh.extents()
    bb_min, bb_max = np.array(bb_min), np.array(bb_max)
    total_size = (bb_max - bb_min).max()

    # Set the center (although this should usually be the origin already).
    centers = (
        (bb_min[0] + bb_max[0]) / 2,
        (bb_min[1] + bb_max[1]) / 2,
        (bb_min[2] + bb_max[2]) / 2
    )
    # Scales all dimensions equally.
    scale = total_size / (1 - padding)

    translation = (
        -centers[0],
        -centers[1],
        -centers[2]
    )
    scales_inv = (
        1/scale, 1/scale, 1/scale
    )

    print(translation)
    print(scales_inv)

    T = np.eye(4)
    trn = np.array([[translation[0]], [translation[1]], [translation[2]]])
    T[:3, 3:] = (1/scale) * trn
    T[:3, :3] = (1/scale) * T[:3, :3]
    print(T)

    mesh.translate(translation)
    mesh.scale(scales_inv)

    mesh.to_obj(output_mesh_path)
    np.savetxt(output_transformation_txt, T)


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Resize op start. Local time is %s" % (local_time_str))


    parser = argparse.ArgumentParser(description='Scale mesh.')
    parser.add_argument('--input_mesh_path', type=str,
                        help='path of input mesh file')
    parser.add_argument('--output_mesh_path', type=str,
                        help='path of output scaled mesh file')
    parser.add_argument('--output_transformation_txt', type=str,
                        help='output transformation txt used in scale')
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Padding applied to the sides (in total).')

    args = parser.parse_args()

    input_mesh_path = args.input_mesh_path
    output_mesh_path = args.output_mesh_path
    output_transformation_txt = args.output_transformation_txt
    padding = args.padding

    run_scale(input_mesh_path=input_mesh_path,
              output_mesh_path=output_mesh_path,
              output_transformation_txt=output_transformation_txt,
              padding=padding)



    t_end = time.time()
    local_time = time.localtime(t_end)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Resize op step done. Local time is %s" % (local_time_str))