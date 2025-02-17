import torch
import numpy as np
import taichi as ti
import trimesh as tm

try:
    from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import Textures
except:
    pass

try:
    import open3d as o3d
except:
    pass


class Mesh:
    def __init__(self, mesh_path, io_type="pytorch3d"):
        self.mesh_path = mesh_path
        self.mesh = None
        self.io_type = io_type
        if self.io_type == "trimesh":
            self.mesh = tm.load(self.mesh_path, force='mesh')
        elif self.io_type == "open3d":
            self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        elif self.io_type == "pytorch3d":
            self.mesh = load_objs_as_meshes([self.mesh_path])
        else:
            raise NotImplementedError(f"IO type {io_type} not implemented.")
        print(f"[{io_type}] Load mesh from {self.mesh_path}" )


    def apply_scale(self, val):
        if self.io_type == "trimesh":
            self.mesh.apply_scale(val)
        elif self.io_type == "pytorch3d":
            verts = self.mesh.verts_list()[0]
            self.mesh.scale_verts_(torch.ones_like(verts) * val)


    def apply_translation(self, offset):
        if self.io_type == "trimesh":
            self.mesh.apply_translation(offset)
        elif self.io_type == "pytorch3d":
            self.mesh.offset_verts_(torch.tensor(offset))


    @property
    def is_watertight(self):
        if self.io_type == "trimesh":
            return self.mesh.is_watertight
        elif self.io_type == "open3d":
            raise Exception
        else:
            raise Exception
        
    @property
    def bounds(self):
        if self.io_type == "trimesh":
            return self.mesh.bounds
        elif self.io_type == "open3d":
            raise Exception
        elif self.io_type == "pytorch3d":
            return self.mesh.verts_list()[0].min(dim=0)[0].numpy(), self.mesh.verts_list()[0].max(dim=0)[0].numpy()
        else:
            raise Exception
        
    @property
    def vertices(self):
        if self.io_type == "trimesh":
            return self.mesh.vertices
        elif self.io_type == "open3d":
            return np.asarray(self.vertices)
        elif self.io_type == "pytorch3d":
            return self.mesh.verts_list()[0].numpy()
        else:
            raise Exception
    
    @property
    def edges(self):
        if self.io_type == "trimesh":
            return self.mesh.edges
        elif self.io_type == "open3d":
            return np.asarray(self.edges)
        elif self.io_type == "pytorch3d":
            return self.mesh.edges_packed().numpy()
        else:
            raise Exception
    
    @property
    def faces(self):
        if self.io_type == "trimesh":
            return self.mesh.faces
        elif self.io_type == "open3d":
            return np.asarray(self.faces)
        elif self.io_type == "pytorch3d":
            return self.mesh.faces_list()[0].numpy()
        else:
            raise Exception


@ti.data_oriented
class SMPLMesh:
    def __init__(self, smpl_mesh) -> None:
        self.raw_mesh = smpl_mesh
        self.num_verts = smpl_mesh.vertices.shape[0]
        self.num_faces = smpl_mesh.faces.shape[0]
        self.faces = ti.field(int, shape=(self.num_faces, 3))
        self.x = ti.field(ti.math.vec3, shape=self.num_verts)

        self.faces.from_numpy(smpl_mesh.faces)
        self.x.from_numpy(smpl_mesh.vertices)

        self.indices = ti.field(dtype = ti.u32, shape = self.num_faces * 3)
        self.initIndices()

    @ti.kernel
    def initIndices(self):
        for idx in range(self.num_faces):
            self.indices[idx * 3 + 0] = self.faces[idx, 0]
            self.indices[idx * 3 + 1] = self.faces[idx, 1]
            self.indices[idx * 3 + 2] = self.faces[idx, 2]


def write_to_obj_o3d(vertices, faces, output_path):
    # translated_verts_ref = verts_ref -  np.min(verts_ref, axis=0)
    # normlized_verts_ref = translated_verts_ref/ np.max(translated_verts_ref, axis=0)
    # color_ref = normlized_verts_ref
    # print("color ref shape ", color_ref.shape)
    output_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    output_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # output_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    output_mesh.triangles = o3d.utility.Vector3iVector(faces)
    # output_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh( f"{output_path}", output_mesh)
    print(f"[open3d] write output mesh to {output_path} done. ")


def write_to_obj_pytorch3d(vertices, faces, output_path):
    # Texture image
    texture_image = torch.ones(256, 256, 3, dtype=torch.float32) * 0.5  # 灰色纹理
    save_obj(output_path, vertices, faces, texture_map=texture_image)
    print(f"[pytorch3d] write output mesh to {output_path} done. ")


def transfer_texture_pytorch3d(output_path, src_mesh, vertices, faces, dst_mesh_path=None):
    if dst_mesh_path:
        dst_mesh = load_objs_as_meshes([dst_mesh_path])
        verts = dst_mesh.verts_list()[0]
        faces = dst_mesh.faces_list()[0]
    else:
        verts = vertices
        faces = faces
    
    # Flip the normal
    faces = faces[:, [0, 2, 1]]

    # Get uvs from src textured mesh
    src_mesh_with_texture = load_objs_as_meshes([src_mesh], load_textures=True)
    faces_uvs = src_mesh_with_texture.textures.faces_uvs_list()[0]
    faces_uvs = faces_uvs[:, [0, 2, 1]] # Flip the uvs normal
    verts_uvs = src_mesh_with_texture.textures.verts_uvs_list()[0]
    texture_map = src_mesh_with_texture.textures.maps_padded()

    save_obj(output_path, verts, faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture_map=texture_map[0])
    print(f"[pytorch3d] write textured output mesh to {output_path} done. ")