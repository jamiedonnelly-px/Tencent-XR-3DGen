import bpy, argparse, sys, os
from pdb import set_trace as st
import time

class Timer:
    _level = 0
    _logs = []

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        Timer._level += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        indent = ' ' * (Timer._level - 1) * 4
        Timer._logs.append(f"{indent}{self.name} took {self.elapsed_time:.3f} seconds")
        Timer._level -= 1

    @staticmethod
    def print_log():
        for log in Timer._logs:
            print(log)
        Timer._logs = []  # Reset logs for next usage 



def remesh_and_bake(object, target_faces=3000, adaptive_size=90, tex_resolution=1024, **kwargs):
    
    bpy.ops.object.select_all(action='DESELECT')
    object.select_set(True)
    bpy.context.view_layer.objects.active = object
    
    # merge seams due to uv unwrapping
    with Timer("merge uv seam"):
        merge_threshold = 1e-3
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=merge_threshold)
        bpy.ops.object.mode_set(mode='OBJECT')
        
    # remesh to quad
    with Timer("quad remesh"):
        bpy.context.scene.qremesher.adaptive_size = adaptive_size
        bpy.context.scene.qremesher.target_count = abs(target_faces)
        bpy.context.scene.qremesher.use_materials = False
        bpy.context.scene.qremesher.use_vertex_color = False
        bpy.context.scene.qremesher.autodetect_hard_edges = True
        bpy.context.scene.qremesher.adapt_quad_count = (target_faces > 0)
        bpy.ops.qremesher.remesh()
        object.hide_set(state=False) 
        
    # un wrap UV
    with Timer("unwrap UV"):
        new_mesh = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=0.78, island_margin=0.01)
        bpy.ops.object.mode_set(mode='OBJECT')

    # create new texture image and material
    with Timer("create new material"):
        new_mesh = bpy.context.view_layer.objects.active
        new_mat = bpy.data.materials.new(name='BakedMaterial')
        new_mesh.data.materials.clear()
        new_mesh.data.materials.append(new_mat)    
        new_mat.use_nodes = True
        bsdf = new_mat.node_tree.nodes.get('Principled BSDF')
        tex_image = new_mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.new('BakedTextureImg', width=tex_resolution, height=tex_resolution)
        new_mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # create cage object for baking
    with Timer("create cage object"):
        new_mesh = bpy.context.view_layer.objects['Retopo_mesh']
        object = bpy.context.view_layer.objects['mesh']
        bpy.context.view_layer.objects.active = new_mesh
        bpy.ops.object.select_all(action='DESELECT')
        new_mesh.select_set(True)
        bpy.ops.object.duplicate(linked=False, mode='TRANSLATION')
        cage_object = bpy.context.active_object
        cage_object.name = f"{new_mesh.name}_cage"
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.transform.shrink_fatten(value=0.05, use_even_offset=False)
        bpy.ops.object.mode_set(mode='OBJECT')

    # bake texture
    with Timer("texture baking"):
        new_mesh.select_set(True)
        object.select_set(True)
        bpy.context.view_layer.objects.active = new_mesh
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 32
        bpy.context.scene.cycles.bake_type = 'DIFFUSE'
        bpy.context.scene.render.bake.use_pass_direct = False
        bpy.context.scene.render.bake.use_pass_indirect = False
        bpy.context.scene.render.bake.use_pass_color = True
        bpy.context.scene.cycles.use_preview_denoising = True
        bpy.context.scene.render.bake.use_selected_to_active = True
        bpy.context.scene.render.bake.use_cage = True
        bpy.context.scene.render.bake.cage_object = cage_object
        cage_object.hide_set(state=True) 
        bpy.ops.object.bake(type='DIFFUSE', use_selected_to_active=True, target='IMAGE_TEXTURES', save_mode='INTERNAL', margin=16, use_clear=True, uv_layer=new_mesh.data.uv_layers.active.name)
    
    object.select_set(False)
    object.hide_set(state=True) 
    
    return new_mesh
    
def load_object(filepath):
    if filepath.lower().endswith('.obj'):
        try:
            bpy.ops.wm.obj_import(filepath=filepath) # v3.0 and above
        except:
            bpy.ops.import_scene.obj(filepath=filepath)  # pre v3.0
    elif filepath.lower().endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif filepath.lower().endswith('.fbx'):
        bpy.ops.import_scene.fbx(filepath=filepath)
        
    imported_object = bpy.context.selected_objects[0]
    imported_object.name = "mesh"

def save_object(filepath):
    if filepath.lower().endswith('.obj'):
        try:
            bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True, export_materials=True) # v3.0 and above
        except:
            bpy.ops.export_scene.obj(filepath=filepath, use_selection=True, use_materials=True) # pre v3.0
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.sequencer_colorspace_settings.name = 'Linear Rec.709'
        bpy.data.images['BakedTextureImg'].save(filepath=os.path.join(os.path.dirname(filepath), 'kd.png'))
        with open(filepath[:-3]+"mtl", "a") as mtl:
            mtl.write("map_Kd kd.png")
    elif filepath.lower().endswith('.glb'):
        bpy.ops.export_scene.gltf(filepath=filepath, use_selection=True)
    elif filepath.lower().endswith('.fbx'):
        bpy.data.images['BakedTextureImg'].pack()
        bpy.ops.export_scene.fbx(filepath=filepath, use_selection=True, path_mode='COPY', embed_textures=True)


if __name__ == '__main__':
    
    argv = sys.argv
    
    raw_argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Remesh an input triangle mesh to quad mesh and bake textures.')
    parser.add_argument('--source_mesh_path', type=str,
                        help='path to source mesh, must be textured. supports glb/obj/fbx')
    parser.add_argument('--destination_mesh_path', type=str,
                        help='path to destination quad mesh. supports glb/obj/fbx')
    parser.add_argument('--target_faces', type=int, default=3000,
                        help='number of quad faces in target mesh')
    parser.add_argument('--adaptive_size', type=float, default=0.9,
                        help='float number that controls how elongated quads can be. 0 for squares, 1 for stripes')
    parser.add_argument('--tex_resolution', type=int, default=1024,
                        help='output mesh texture resolution')
    
    args = parser.parse_args(raw_argv)

    # clear data and load
    with Timer("load mesh"):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        load_object(args.source_mesh_path)

    with Timer("remesh & bake"):
        new_obj = remesh_and_bake(bpy.context.selected_objects[0], args.target_faces, int(100*args.adaptive_size), args.tex_resolution)
    
    # save
    with Timer("export mesh"):
        bpy.ops.object.select_all(action='DESELECT')
        new_obj.select_set(True)
        bpy.context.view_layer.objects.active = new_obj
        save_object(args.destination_mesh_path)
    
    print("========running time========")
    Timer.print_log()
    
    bpy.ops.wm.quit_blender()