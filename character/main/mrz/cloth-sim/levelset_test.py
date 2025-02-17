import numpy as np
import trimesh as tm
import taichi as ti
from levelset import LevelSetSdfModel

ti.init(arch=ti.cuda, kernel_profiler=False, device_memory_GB=4)
vec3 = ti.types.vector(3, ti.f32)

mesh = tm.load("D:\\workspace\\cloth-sim\\smpl_deformed_shirt_fix_opt_0000.obj", force='mesh')
mesh_scale = 1

mesh.apply_scale(mesh_scale)
if np.any(mesh.bounds[1]-mesh.bounds[0] > 1):
    mesh_scale = np.max(mesh.bounds[1]-mesh.bounds[0]) #* 1.1

mesh.apply_scale(1.0/mesh_scale)
mesh.apply_translation(-mesh.bounds[0])
print("bounds:", mesh.bounds)

# res = mesh.fill_holes()
print("watertight:", mesh.is_watertight)

mesh_scale = 10
mesh.apply_scale(mesh_scale)

model_vertices_np = mesh.vertices
model_indices_np = mesh.faces.flatten()

model_vertices = ti.Vector.field(3, ti.f32, model_vertices_np.shape[0])
model_indices = ti.field(ti.i32, model_indices_np.shape[0])
model_vertices.from_numpy(model_vertices_np)
model_indices.from_numpy(model_indices_np)

model_vertices_temp = ti.Vector.field(3, ti.f32, model_vertices_np.shape[0])
model_indices_temp = ti.field(ti.i32, model_indices_np.shape[0])
model_vertices_temp.from_numpy(model_vertices_np)
model_indices_temp.from_numpy(model_indices_np)
sdf_resolution = 256
sdf = LevelSetSdfModel(model_vertices_temp, model_indices_temp, 1.0/mesh_scale, vec3(0.0), sdf_resolution)

# self.mesh_scale = mesh_scale

# self.set_particle_radius(0.008)

# self.sdf = sdf
# self.mesh_scale = mesh_scale

# self.boundary_box[0] = vec3(0.0, -0.0, 0.0)
# self.boundary_box[1] = vec3(1.0, 1.0, 1.0) * mesh_scale

# self.time_step = 1.0/30 / 8
# self.pbd_iter_n = 6
# # self.fluid_pressure_k = 0.00002
# self.fluid_pressure_k = 1e-6