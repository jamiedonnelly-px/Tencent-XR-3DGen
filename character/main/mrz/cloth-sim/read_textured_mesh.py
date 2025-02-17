import torch
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures


mesh_path = "/Users/mingruizhang/workspace/cloth-sim/5b8c/results/textured.obj"
mesh = load_objs_as_meshes([mesh_path], load_textures=True)

fitted_mesh_path = "/Users/mingruizhang/workspace/cloth-sim/fit_new_garment_deformed_correspondence_test_cloth.obj"
fitted_mesh = load_objs_as_meshes([fitted_mesh_path])

verts = fitted_mesh.verts_list()[0]
faces = fitted_mesh.faces_list()[0]
faces = faces[:, [0, 2, 1]]

faces_uvs = mesh.textures.faces_uvs_list()[0]
faces_uvs = faces_uvs[:, [0, 2, 1]]

verts_uvs = mesh.textures.verts_uvs_list()[0]

texture_map = mesh.textures.maps_padded()
save_obj("output_with_texture.obj", verts, faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture_map=texture_map[0])
