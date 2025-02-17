import os, bpy, bmesh, mathutils, shutil

names = {
    "smplx_mesh": "SMPLX-mesh-female",
    "smplx_armature": "SMPLX-female", 
    "armature": "mesh_armature",
    "head": "head_mesh",
    "body": "body_mesh"
}

from bpy_extras import object_utils

class fix_mesh_transparency(object):
    def toggle_alpha_blend_mode(obj, blend_method='OPAQUE'):
        if obj.material_slots:
            for slot in obj.material_slots:
                if slot.material:
                    slot.material.blend_method = blend_method
                
    def remove_alpha_image(obj):
        if obj.material_slots:
            for slot in obj.material_slots:
                if slot.material:
                    node_tree = slot.material.node_tree
                    for node in node_tree.nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            if len(node.inputs["Alpha"].links) > 0:
                                l = node.inputs["Alpha"].links[0]
                                if l is not None:
                                    alpha_image_node = l.from_node
                                    if alpha_image_node is not None:
                                        alpha_image = alpha_image_node.image
                                        node_tree.nodes.remove(alpha_image_node)
                                        bpy.data.images.remove(alpha_image)
                                    # node_tree.links.remove(l)
                                    node.inputs["Alpha"].default_value = 1.0
                    
    def toggle_alpha_linkage(object):
        if object.material_slots:
            for slot in object.material_slots:
                if slot.material:
                    node_tree = slot.material.node_tree
                    for node in node_tree.nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            if len(node.inputs["Alpha"].links) > 0:
                                l = node.inputs["Alpha"].links[0]
                                node.inputs["Alpha"].links.new(
                                    l.input.outputs["Alpha"], node.inputs["Alpha"])
                    
    def alter_image_node_path(object, new_image_folder: str):
        if object.material_slots:
            for slot in object.material_slots:
                if slot.material:
                    material_name = slot.material.name
                    node_tree = slot.material.node_tree
                    for node in node_tree.nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            if len(node.inputs["Base Color"].links) > 0:
                                l = node.inputs["Base Color"].links[0]
                                if l is not None:
                                    original_tex_image_node = l.from_node
                                    old_image_name = original_tex_image_node.image.filepath
                                    old_image_filename = os.path.split(
                                        old_image_name)[1]
                                    new_image_name = os.path.join(
                                        new_image_folder, material_name+"_"+old_image_filename)
                                    
                                    shutil.copyfile(old_image_name, new_image_name)

                                    texture_image = bpy.data.images.load(
                                        new_image_name)
                                    diffusion_node = node_tree.nodes.new(
                                        "ShaderNodeTexImage")
                                    diffusion_node.image = texture_image
                                    diffusion_node.image.colorspace_settings.name = "sRGB"

                                    if l is not None:
                                        node_tree.links.remove(l)
                                    if original_tex_image_node is not None:
                                        node_tree.nodes.remove(
                                            original_tex_image_node)

                                    node_tree.links.new(
                                        diffusion_node.outputs["Color"], node.inputs["Base Color"])
                                    
    @staticmethod               
    def run(out_textured_path):
        bpy.ops.object.select_all(action='DESELECT')
        for int, obj in enumerate(bpy.context.scene.objects):
            if obj.type == 'MESH':
                fix_mesh_transparency.remove_alpha_image(obj)
                fix_mesh_transparency.toggle_alpha_blend_mode(obj)
                fix_mesh_transparency.toggle_alpha_linkage(obj)
                fix_mesh_transparency.alter_image_node_path(obj, out_textured_path)

class base_func(object):
    @staticmethod
    def calculate_polygon_normal(vertices):
        edge1 = vertices[1].co - vertices[0].co
        edge2 = vertices[2].co - vertices[1].co
        normal = edge1.cross(edge2)
        normal.normalize()
        return normal
    
    @staticmethod
    def set_mesh_original():
        for mesh in bpy.context.scene.objects:
            if mesh.type == 'MESH':
                mesh_obj = bpy.data.objects[mesh.name]
                bm = bmesh.new()
                bm.from_mesh(mesh_obj.data)
                for vertex in bm.verts:
                    vertex.co = mesh_obj.matrix_world @ vertex.co
                bm.to_mesh(mesh_obj.data)
                bm.free()
                mesh_obj.matrix_world = mathutils.Matrix.Identity(4)

    @staticmethod
    def bind_mesh_armature(obj_mesh, armature):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action='DESELECT')
        obj_mesh.select_set(True)
        bpy.context.view_layer.objects.active = armature
        obj_mesh.parent = armature
        obj_mesh.parent_type = 'ARMATURE'
        armature_modifier = obj_mesh.modifiers.new(name="Armature", type="ARMATURE")
        armature_modifier.object = armature
    

class utils(object):
    @staticmethod
    def select(obj_names):
        bpy.ops.object.select_all(action='DESELECT') 
        for obj_name in obj_names:
            obj = bpy.data.objects[obj_name]
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj 

    @staticmethod
    def delete_all():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False) 

    @staticmethod
    def load_obj_mesh(mesh_path: str, z_up=False):
        print("test", mesh_path)
        bpy.ops.object.select_all(action='DESELECT')
        version_info = bpy.app.version
        if version_info[0] > 2:
            if z_up:
                bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis='Y', up_axis='Z')
            else:
                bpy.ops.wm.obj_import(filepath=mesh_path, forward_axis='NEGATIVE_Z', up_axis='Y')
        else:
            bpy.ops.import_scene.obj(
                filepath=mesh_path, axis_forward='-Z', axis_up='Y')
        meshes = []
        for ind, obj in enumerate(bpy.context.selected_objects):
            if obj.type == 'MESH':
                meshes.append(obj)
        return meshes
    
    @staticmethod
    def import_file(file_path: str):
        if not os.path.exists(file_path):
            print("file is not exists! file: ", file_path)
            return
        if file_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=file_path)
        elif file_path.endswith(".obj"):
            bpy.ops.import_scene.obj(filepath=file_path)
            # utils.load_obj_mesh(file_path)

        elif file_path.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=file_path)

    @staticmethod
    def export_fbx(file_path: str, obj_names=None):
        use_selection_flag = False
        if obj_names:
            bpy.ops.object.select_all(action='DESELECT')
            utils.select(obj_names)
            use_selection_flag = True
        bpy.ops.export_scene.fbx(filepath=file_path, use_selection=use_selection_flag, embed_textures=True, path_mode='COPY')    