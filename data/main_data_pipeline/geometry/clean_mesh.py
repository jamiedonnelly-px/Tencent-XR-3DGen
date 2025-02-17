import argparse
import time

import numpy as np
import pymeshlab
import trimesh

try:
    from pymeshlab import Percentage as Percentage
except:
    from pymeshlab import PercentageValue as Percentage


def clean_mesh(verts, faces, diameter_percentage: int = 0.1, repair: bool = True, min_isolate: int = 1000,
               unique_mesh: bool = True):
    """Cleaning generated 3D meshes

    :param verts: 3D coordinates of input mesh vetices with shape [N, 3]
    :param faces: 3D coordinates of input mesh faces with shape [N, 3]

    :return verts: 3D coordinates of cleaned mesh vetices with shape [N, 3]
    :return faces: 3D coordinates of cleaned mesh faces with shape [N, 3]
    """

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pymeshlab.Mesh(verts, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces
    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_isolate)

    if diameter_percentage > 0:
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.Percentage(diameter_percentage))

    if unique_mesh:
        ms.generate_splitting_by_connected_components()
        face_count = [m.face_number() for m in ms]
        new_index = np.argmax(face_count[1:]) + 1
        new_mesh = pymeshlab.Mesh(vertex_matrix=ms[new_index].vertex_matrix(), face_matrix=ms[new_index].face_matrix())
        ms = pymeshlab.MeshSet()
        ms.add_mesh(new_mesh)

    if repair:
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f"Finish mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}")

    return verts, faces


def export_obj(verts: np.ndarray, faces: np.ndarray, filename: str) -> None:
    """
    Exports vertices and faces as an OBJ file.

    Parameters:
    - verts: numpy array of shape [n_verts, 3], representing the vertices.
    - faces: numpy array of shape [n_faces, 3], representing the face indices (0-based).
    - filename: str, the name of the output OBJ file.
    """
    with open(filename, 'w') as file:
        # Write vertices
        for vert in verts:
            file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

        # Write faces (OBJ format is 1-based indexing)
        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


if __name__ == '__main__':
    t_start = time.time()
    local_time = time.localtime(t_start)
    local_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
    print("Clean up mesh starts. Local time is %s" % (local_time_str))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mesh_path", type=str, default="",
                        help="path to manifold obj")
    parser.add_argument("--output_mesh_path", type=str, default="",
                        help="output path of sampled points")
    parser.add_argument('--minimal_face_number', type=int, default=1000,
                        help='isolated pieces with number lower than this will be removed')
    parser.add_argument('--diameter_percentage', type=int, default=-1,
                        help='pieces smaller than this diameter will be removed')
    args = parser.parse_args()

    mesh_path = args.mesh_path
    output_mesh_path = args.output_mesh_path
    diameter_percentage = args.diameter_percentage
    minimal_face_number = args.minimal_face_number
    original_mesh = trimesh.load(mesh_path)

    modified_verts, modified_faces = clean_mesh(original_mesh.vertices,
                                                original_mesh.faces,
                                                diameter_percentage=diameter_percentage,
                                                min_isolate=minimal_face_number)

    export_obj(modified_verts, modified_faces, output_mesh_path)
