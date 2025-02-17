import bpy, argparse, sys, os
import math
from pdb import set_trace as st
from math import radians
import time

def mesh_shade_smooth(object):
    version_info = bpy.app.version
    print(version_info)
    if version_info[0] >= 4 and version_info[1] >= 1:
        node = object.modifiers.new("Smooth by Angle", "NODES")
        result = bpy.ops.object.modifier_add_node_group(asset_library_type='ESSENTIALS', asset_library_identifier="",
                                                        relative_asset_identifier="geometry_nodes\\smooth_by_angle.blend\\NodeTree\\Smooth by Angle")
        if 'CANCELLED' in result:
            return

        modifier = object.modifiers[-1]
        modifier["Socket_1"] = True
        modifier["Input_1"] = math.radians(30)
        object.update_tag()
    else:
        object.data.use_auto_smooth = True
        object.data.auto_smooth_angle = math.radians(30)
    for f in object.data.polygons:
        f.use_smooth = True
    
def remesh_and_bake(object, target_faces=3000, adaptive_size=90, tex_resolution=1024, quad_use_normals = False, **kwargs):
    bpy.ops.object.select_all(action='DESELECT')
    object.select_set(True)
    bpy.context.view_layer.objects.active = object
    
    time_list = []
    ts = time.time()
    # merge seams due to uv unwrapping
    merge_threshold = 1e-3
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_threshold)
    bpy.ops.object.mode_set(mode='OBJECT')
    time_list.append(("remove_doubles", time.time() -  ts))
        
    ts = time.time()
    # remesh to quad
    # use auto normal to guide remesh
    if quad_use_normals:
        try:
            bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=0.349066) # blender 4.2
        except:
            try:
                bpy.ops.object.shade_smooth_by_angle(angle=0.349066, keep_sharp_edges=False) # blender 4.1
            except:
                bpy.ops.object.shade_smooth(use_auto_smooth=True, auto_smooth_angle=0.349066) # older version
        bpy.context.scene.qremesher.use_normals = True

    bpy.context.scene.qremesher.adaptive_size = adaptive_size
    bpy.context.scene.qremesher.target_count = abs(target_faces)
    bpy.context.scene.qremesher.use_materials = False
    bpy.context.scene.qremesher.use_vertex_color = False
    bpy.context.scene.qremesher.autodetect_hard_edges = True
    bpy.context.scene.qremesher.adapt_quad_count = (target_faces > 0)
    bpy.ops.qremesher.remesh()
    object.hide_set(state=False) 
    time_list.append(("qremesher", time.time() -  ts))
        
    # un wrap UV
    ts = time.time()
    new_mesh = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=0.78, island_margin=0.01)
    bpy.ops.object.mode_set(mode='OBJECT')
    time_list.append(("uv", time.time() -  ts))
    
    ### 
    ts = time.time()
    mesh_shade_smooth(new_mesh)
    time_list.append(("mesh_shade_smooth", time.time() -  ts))

    ts = time.time()
    # create new texture image and material
    new_mesh = bpy.context.view_layer.objects.active
    new_mat = bpy.data.materials.new(name='BakedMaterial')
    new_mesh.data.materials.clear()
    new_mesh.data.materials.append(new_mat)    
    new_mat.use_nodes = True
    bsdf = new_mat.node_tree.nodes.get('Principled BSDF')
    tex_image = new_mat.node_tree.nodes.new('ShaderNodeTexImage')

    if kwargs.get("geom_only", False):
        print('[TESTINFO] geom_only')
        fake_res = 64
        tex_image.image = bpy.data.images.new('BakedTextureImg', width=fake_res, height=fake_res)
        # Set the texture image to a gray color
        gray_color = (0.7612, 0.7612,0.7612, 1.0)
        tex_image.image.pixels = gray_color * fake_res * fake_res
        tex_image.image.update()
        
        new_mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

        # Set the material properties
        new_mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.677
        new_mat.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
        new_mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.223
        new_mat.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = 1.0
        time_list.append(("add_tex", time.time() -  ts))
        print('[INFO] remesh_and_bake time_list: ', time_list)
                
        return new_mesh

    
    tex_image.image = bpy.data.images.new('BakedTextureImg', width=tex_resolution, height=tex_resolution)
    new_mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # create cage object for baking
    
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

    #### 
    bpy.context.scene.view_settings.view_transform = 'Standard'
    
    # bake texture
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if filepath.lower().endswith('.obj'):
        try:
            bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True, export_materials=True) # v3.0 and above
        except:
            bpy.ops.export_scene.obj(filepath=filepath, use_selection=True, use_materials=True) # pre v3.0
        try:
            bpy.context.scene.view_settings.view_transform = 'Standard'
            bpy.context.scene.sequencer_colorspace_settings.name = 'Linear Rec.709'
        except:
            print('warn faild sequencer_colorspace_settings when export obj. todo')
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
    parser.add_argument('--geom_only', action="store_true",
                        help='skip bake if geom only')
    args = parser.parse_args(raw_argv)

    # clear data and load
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    load_object(args.source_mesh_path)

    new_obj = remesh_and_bake(bpy.context.selected_objects[0], args.target_faces, int(100*args.adaptive_size), args.tex_resolution, geom_only=args.geom_only)
    
    # save
    bpy.ops.object.select_all(action='DESELECT')
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj
    save_object(args.destination_mesh_path)
    if os.path.splitext(args.destination_mesh_path)[1] != ".obj":
        out_obj_path = os.path.splitext(args.destination_mesh_path)[0] + ".obj"
        save_object(out_obj_path)
    
    bpy.ops.wm.quit_blender()