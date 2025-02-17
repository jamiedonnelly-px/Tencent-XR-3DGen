import taichi as ti
import numpy as np
from render import *
from solver import PositionBasedDynamics
from sdf import *
import argparse
import os
from levelset import LevelSetSdfModel
from mesh_utils import Mesh, SMPLMesh, write_to_obj_o3d, write_to_obj_pytorch3d, transfer_texture_pytorch3d

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='cpu')
parser.add_argument('--smpl_mesh', required=True)
parser.add_argument('--garment_mesh', required=True)
parser.add_argument('--textured_mesh', default="", help="Textured mesh e.g., from syncmvd")
parser.add_argument('--io_type', default='trimesh')      # Use trimesh to load the mesh for simulation, use pytorch3d for export textured mesh
parser.add_argument('--garment_name', default='test_garment')
parser.add_argument('--fit_iter', type=int, default=150) # Iterations for fitting garment on to human body
parser.add_argument('-v', '--interactive', action='store_true') # Iterations for fitting garment on to human body
args = parser.parse_args()

# TODO: remove all possible f64 or i64 because MacOS metal backend doesn't support these types
# a workaround is use ti.cpu backend instead when on MacOS
ARCH = ti.cpu
if args.arch == "gpu":
    ARCH = ti.gpu
ti.init(arch=ARCH, device_memory_GB=2)
vec3 = ti.types.vector(3, ti.f32)

IO_TYPE = args.io_type

# smpl_mesh_path = "smpl_deformed_test_cloth_opt_0000_0100.obj"
smpl_mesh_path = args.smpl_mesh
smpl_mesh = Mesh(mesh_path=smpl_mesh_path, io_type=IO_TYPE)

print(f"SMPL mesh bound: {smpl_mesh.bounds}")
mesh_scale = 1
if np.any(smpl_mesh.bounds[1]-smpl_mesh.bounds[0] > 1):
    mesh_scale = np.max(smpl_mesh.bounds[1]-smpl_mesh.bounds[0]) #* 1.1
print(f"mesh scale： {mesh_scale}， bounds: {smpl_mesh.bounds}")

mesh_scale_to_apply = 1.0/mesh_scale
mesh_trans_to_apply = -smpl_mesh.bounds[0]
smpl_mesh.apply_scale(mesh_scale_to_apply)
print(f"SMPL mesh bound after scale: {smpl_mesh.bounds}")
smpl_mesh.apply_translation(mesh_trans_to_apply)
print(f"SMPL mesh bound after translation: {smpl_mesh.bounds}")


# Construct the SDF based on SMPL model
model_vertices_np = smpl_mesh.vertices
model_indices_np = smpl_mesh.faces.flatten()
model_vertices = ti.Vector.field(3, ti.f32, model_vertices_np.shape[0])
model_indices = ti.field(ti.i32, model_indices_np.shape[0])
model_vertices.from_numpy(model_vertices_np)
model_indices.from_numpy(model_indices_np)

model_vertices_temp = ti.Vector.field(3, ti.f32, model_vertices_np.shape[0])
model_indices_temp = ti.field(ti.i32, model_indices_np.shape[0])
model_vertices_temp.from_numpy(model_vertices_np)
model_indices_temp.from_numpy(model_indices_np)
sdf_resolution = 512
sdf = LevelSetSdfModel(model_vertices_temp, model_indices_temp, 1.0, vec3(0.0), sdf_resolution)


# SMPL mesh wrapper
smpl_mesh_wrapper = SMPLMesh(smpl_mesh=smpl_mesh)

# Load the garment mesh
# garment_mesh_path = "garment_deformed_correspondence_test_cloth.obj"
garment_mesh_path = args.garment_mesh
garment_mesh = Mesh(mesh_path=garment_mesh_path, io_type=IO_TYPE)
garment_mesh.apply_scale(mesh_scale_to_apply)
garment_mesh.apply_translation(mesh_trans_to_apply)

scale_up = 1.00
garment_mesh.apply_scale(scale_up)

stretch_compilance = 1e-7 # smaller -> stiff, larger -> soft
bending_compilance = 1e5  # smaller -> stiff, larger -> soft

solver = PositionBasedDynamics( cloth_mesh = garment_mesh,
                                sdf = sdf,
                                stretch_compliance = stretch_compilance,
                                bending_compliance = bending_compilance,
                                frame_dt = 1e-2, # how many physical time for visualizing one frame
                                dt = 5e-4,       # simulator timestep
                                rest_iter= 5,    # How many iterations for solving each constraint
                                XPBD=True)

if not args.interactive:
    # Non-interactive mode, garment fit use
    # solve by step for 'fit_iter' steps
    name = args.garment_name
    textured_mesh_output_folder = "output_textured_mesh"
    os.makedirs(textured_mesh_output_folder, exist_ok=True)

    is_pytorch3d = True
    try:
        import pytorch3d
    except:
        is_pytorch3d = False
        print("No pytorch3d is found. Use open3d for export, no texture will be attached.")
    if is_pytorch3d:
        write_to_obj_pytorch3d(solver.verts.x.to_torch(), solver.faces.to_torch(), f"{textured_mesh_output_folder}/{name}_no_textured_initial.obj")
        if args.textured_mesh != '':
            transfer_texture_pytorch3d(f"{textured_mesh_output_folder}/{name}_fit_texutred_initial.obj", args.textured_mesh, solver.verts.x.to_torch(), solver.faces.to_torch())
        else:
            print("No corresponding textured mesh is provided.")
    else:
        
        write_to_obj_o3d(solver.verts.x.to_numpy(), solver.faces.to_numpy(), f"{textured_mesh_output_folder}/{name}_no_textured_initial.obj")

    for _ in range(args.fit_iter):
        solver.solve_by_step()
        solver.clear_velocity()
    
    # Here we use the pytorch3D for exporting OBJ
    # The reason is to be consistent to the syncmvd output textured mesh, otherwise the uvs will mess up
    # If texture is not needed, then feel free to use open3d for export

    if is_pytorch3d:
        write_to_obj_pytorch3d(solver.verts.x.to_torch(), solver.faces.to_torch(), f"{textured_mesh_output_folder}/{name}_no_textured.obj")
        write_to_obj_pytorch3d(smpl_mesh_wrapper.x.to_torch(), smpl_mesh_wrapper.faces.to_torch(), f"{textured_mesh_output_folder}/{name}_smpl.obj")
        if args.textured_mesh != '':
            transfer_texture_pytorch3d(f"{textured_mesh_output_folder}/{name}_fit_texutred.obj", args.textured_mesh, solver.verts.x.to_torch(), solver.faces.to_torch())
        else:
            print("No corresponding textured mesh is provided.")
    else:
        write_to_obj_o3d(solver.verts.x.to_numpy(), solver.faces.to_numpy(), f"{textured_mesh_output_folder}/{name}_no_textured.obj")
        write_to_obj_o3d(smpl_mesh_wrapper.x.to_numpy(), smpl_mesh_wrapper.faces.to_numpy(), f"{textured_mesh_output_folder}/{name}_smpl.obj")

else:
    # Interactive mode
    # Vulkan is needed for Taichi GGUI

    # ====== Garment Fit ====== 
    # 1. The garment and SMPL body should be in a roughly fitted position after the previous landmark based fit process
    # 2. Keep pressing "space" on keyboard for continously garment fit
    # 3. When there the garment is a good fit on the SMPL body, click "export_obj" to export the garment and SMPL mesh
    
    # ====== Garment Fit and Simulation ====== 
    # 1. Perform the same procedure as Garment Fit
    # 2. click "ready_for_sim", this is important for setting up a correct initial state for the simulator
    # 3. Then click "run_sim" or press "r" on keyboard for starting the simulation
    
    name = args.garment_name
    textured_mesh_output_folder = "output_textured_mesh"
    os.makedirs(textured_mesh_output_folder, exist_ok=True)

    is_pytorch3d = True
    try:
        import pytorch3d
    except:
        is_pytorch3d = False
        print("No pytorch3d is found. Use open3d for export, no texture will be attached.")


    viz = Visualizer()
    window = viz.initScene(
            position=(0.4, -0.8, 0.925), 
            lookat=(0.52, 0.52, 0.4), 
            show_window=True)

    frame = 0
    running = True
    cloth_fit_iter = 0
    while running:
        if window.is_pressed(" "):
            viz.cloth_fit = True
        if window.is_pressed("r"):
            viz.run_sim = True
        x_offset = 0.0
        y_offset = 0.0
        z_offset = 0.0
        move_speed = 0.0025
        if window.is_pressed("j"):
            # left
            x_offset -= move_speed
        if window.is_pressed("k"):
            # down
            z_offset -= move_speed
        if window.is_pressed("l"):
            # right
            x_offset += move_speed
        if window.is_pressed("i"):
            # up
            z_offset += move_speed
        if window.is_pressed("u"):
            # forward
            y_offset += move_speed
        if window.is_pressed("o"):
            # back
            y_offset -= move_speed
        if viz.cloth_fit:
            solver.solve_by_step()
            solver.clear_velocity()
            viz.cloth_fit = False
            cloth_fit_iter += 1

        if viz.ready_for_sim:
            solver.set_current_state_as_rest_state()
            viz.ready_for_sim = False

        if viz.run_sim:
            solver.solve()
        if viz.export_obj:
            if is_pytorch3d:
                write_to_obj_pytorch3d(solver.verts.x.to_torch(), solver.faces.to_torch(), f"{textured_mesh_output_folder}/{name}_no_textured.obj")
                write_to_obj_pytorch3d(smpl_mesh_wrapper.x.to_torch(), smpl_mesh_wrapper.faces.to_torch(), f"{textured_mesh_output_folder}/{name}_smpl.obj")
                if args.textured_mesh != '':
                    transfer_texture_pytorch3d(f"{textured_mesh_output_folder}/{name}_fit_texutred.obj", args.textured_mesh, solver.verts.x.to_torch(), solver.faces.to_torch())
                else:
                    print("No corresponding textured mesh is provided.")
            else:
                write_to_obj_o3d(solver.verts.x.to_numpy(), solver.faces.to_numpy(), f"{textured_mesh_output_folder}/{name}_no_textured.obj")
                write_to_obj_o3d(smpl_mesh_wrapper.x.to_numpy(), smpl_mesh_wrapper.faces.to_numpy(), f"{textured_mesh_output_folder}/{name}_smpl.obj")

            viz.export_obj = False
            print(f"Saved garment the after {cloth_fit_iter} ")
        solver.translation(ti.Vector([x_offset, y_offset, z_offset]))
        solver.covert_sdf_color()
        running = viz.renderScene(solver, smpl_mesh, frame)
        frame += 1
