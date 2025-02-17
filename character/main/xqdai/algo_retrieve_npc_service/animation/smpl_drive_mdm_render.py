import os
import bpy
import glob
from pathlib import Path
from numpy import arange, pi, sin, cos, arccos
import bmesh
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler

pi = 3.14


def sphere_point_sample(n = 300):
    # use fibonacci spiral
    goldenRatio = (1 + 5 ** 0.5) / 2
    i = arange(0, n)
    theta = 2 * pi * i / goldenRatio
    phi = arccos(1 - 2 * (i + 0.5) / n)
    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    return np.stack ([x,y,z], axis=-1)

def opencv_to_blender(T):
    """T: ndarray 4x4
       usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
    """
    origin = np.array(((1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0,  0, 1)))
    return np.matmul(T,origin) #T * origin

def blender_to_opencv(T):
    transform = np.array(((1, 0, 0, 0),
              (0, -1, 0, 0),
              (0, 0, -1, 0),
              (0, 0, 0, 1)))
    return np.matmul(T,transform)#T * transform


def look_at(obj_camera, point):
    loc_camera = obj_camera.location
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


def set_camera( bpy_cam,  angle=pi / 3, W=600, H=500):
    bpy_cam.angle = angle
    bpy_scene = bpy.context.scene
    bpy_scene.render.resolution_x = W
    bpy_scene.render.resolution_y = H

def read_mesh_to_ndarray( mesh, mode = "Edit"):
    ''' read the vert coordinate of a deformed mesh
    :param mesh: mesh object
    :return: numpy array of the mesh
    '''
    assert mode in [ "edit", "object"]

    if mode is "object" :
        bm = bmesh.new()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm.from_object(mesh, depsgraph)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray( mverts_co, dtype=np.float32)
        # faces = [[ v.index for v in face.verts ] for face in bm.faces]
        # print(f"face length:{len(faces)}")
        # print(faces)
        # faces=np.asarray(faces,dtype=np.int32)
        bm.free()
    elif mode is "edit" :
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.editmode_toggle()
        bm = bmesh.from_edit_mesh(mesh.data)
        mverts_co = [(v.co) for v in bm.verts]
        mverts_co = np.asarray( mverts_co, dtype=np.float32)
        # faces = [[ v.index for v in face.verts ] for face in bm.faces]
        # faces=np.asarray(faces,dtype=np.int32)
        bm.free()
        bpy.ops.object.editmode_toggle()

    return mverts_co, []

def compute_mesh_size( meshes ) :
    verts = []
    for ind, mesh in enumerate(meshes):
        vert, _ = read_mesh_to_ndarray( mesh, mode="object")
        mat = np.asarray(mesh.matrix_world)
        R,t = mat[:3,:3], mat[:3,3:] #Apply World Scale
        verts.append( ( R@vert.T + t ).T )
    verts=np.concatenate(verts, axis=0)
    obj_center = verts.mean(axis=0)
    min_ = verts.min(axis=0)
    max_ = verts.max(axis=0)
    length = np.linalg.norm(max_ - min_)

    print( "object mode ", obj_center, length)


    #计算角色中心点,身体长度
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    verts = []
    for ind, mesh in enumerate(meshes):
        print(ind)
        vert, _ = read_mesh_to_ndarray( mesh, mode="edit")
        mat = np.asarray(mesh.matrix_world)
        R,t = mat[:3,:3], mat[:3,3:] #Apply World Scale
        verts.append( ( R@vert.T + t ).T )
    verts=np.concatenate(verts, axis=0)
    obj_center = verts.mean(axis=0)
    min_ = verts.min(axis=0)
    max_ = verts.max(axis=0)
    length = np.linalg.norm(max_ - min_)

    print( "edit mode ", obj_center, length)

    return obj_center, length

def get_largest_armature():
    largest_armature = None
    max_vertex_count = 0

    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            vertex_count = len(obj.data.bones)
            if vertex_count > max_vertex_count:
                max_vertex_count = vertex_count
                largest_armature = obj

    return largest_armature


def file_import(character_path):
    data_formate = character_path[-3:]
    if "fbx" == data_formate:
        bpy.ops.import_scene.fbx(filepath=character_path, use_anim=True)  
    elif "glb" == data_formate:
        bpy.ops.import_scene.gltf(filepath=character_path)
    else:
        print("data formate not support")
        return -1
    return 0


def remove_alpha_image(object):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                node_tree = slot.material.node_tree

                for node in node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        if len(node.inputs["Alpha"].links) > 0:
                            l = node.inputs["Alpha"].links[0]
                            if l is not None:
                                node_tree.links.remove(l)
                                node.inputs["Alpha"].default_value = 1.0


def toggle_alpha_blend_mode(object, blend_method='OPAQUE'):
    if object.material_slots:
        for slot in object.material_slots:
            if slot.material:
                slot.material.blend_method = blend_method

if __name__ == '__main__':
    import sys
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] 
    fbx_file = argv[0]

    current_file_path = os.path.abspath(__file__)
    current_file_directory = os.path.dirname(current_file_path)

    output_folder = current_file_directory+'/save/output_'
    output_folder = output_folder+fbx_file.split("/")[-4]+"_"+fbx_file.split("/")[-3]
    # root_translation = np.load("/apdcephfs/private_xiaqiangdai/workspace/motion/motion-diffusion-model/save/drive_root_translation.npy")
    # root_loc = root_translation[:,0]

    if os.path.exists(output_folder):
        os.system("rm -rf "+output_folder)
    os.system("mkdir "+output_folder)
    weapon = ['weapon']
    engine_type = "eevee"
    device = "GPU"
    use_light = "hdr"
    H, W = 512, 512

    if len(bpy.context.scene.objects)!=0:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
    # 导入FBX文件
    if file_import(fbx_file)!=0:
        print("drive_render:file import error")
        assert(-1)
        
    for mesh in bpy.context.scene.objects:
        if mesh.type == 'MESH':
            toggle_alpha_blend_mode(mesh)


    bpy.ops.object.select_all(action='DESELECT')

    meshes=[]
    for ind,  obj in enumerate( bpy.context.scene.objects):
        if obj.type == 'MESH':
            if any( wp in obj.name for wp in weapon ):
                obj.select_set(state=True)
                bpy.ops.object.delete()
            else :
                meshes.append (obj)
    obj_center, length = compute_mesh_size(meshes)
    print(f"obj_center:{obj_center}")
    print(f"length:{length}")

    bpy.ops.object.select_all(action='DESELECT')

    largest_obj = get_largest_armature()
    bpy.context.view_layer.objects.active = largest_obj
    largest_obj.select_set(state=True)
    print(f"obj_center half:{obj_center[2]/2}")
    bpy.context.object.delta_location[2] = -obj_center[2]/2*1.4


    length2 = length*1.2#2.5
    # scale = length2/length
    # for mesh in meshes:
    #     # mat = np.asarray(mesh.matrix_world)
    #     trn = -1 * obj_center[...,np.newaxis]
    #     T = np.eye(4)
    #     # print ("trn.shape", trn.shape)
    #     # print("T[:3,3:].shape", T[:3,3:].shape)
    #     T[:3,3:] = trn
    #     T = scale * T
    #     mesh.matrix_world = Matrix(T) @ mesh.matrix_world


    print(f"obj_center:{obj_center}")
    #设置Camera，在物体为中心的球面上采样相机
    n_cam = 300
    points = sphere_point_sample(n_cam)
    points = points * length2   #scale
    cam_names = ["cam-%04d" % i for i in range (n_cam)]
    for i in range (n_cam):
        if i!=87:
            continue
        camera_data = bpy.data.cameras.new(name=cam_names[i])
        camera_object = bpy.data.objects.new(cam_names[i], camera_data)
        bpy.context.scene.collection.objects.link(camera_object)
        camera_object.location = Vector (points[i])
        # look_at_point =  Vector((obj_center[0],obj_center[1],obj_center[2]))
        look_at_point =  Vector((0,0,0))
        camera_data.display_size = length2*0.1
        camera_data.clip_start = 0.01
        camera_data.clip_end = 100
        set_camera(camera_data, angle=pi/3, W=W, H=H)
        look_at(camera_object, look_at_point)
        bpy.context.view_layer.update() #update camera params
        
        bpy.context.scene.camera = camera_object


    # 设置光源， 在球面上设置若干点光源
    n_lights = 10
    if use_light == "point":
        lights_center = sphere_point_sample(n_lights)
        lights_center = lights_center * standard_height  # scale and translate
        for i in range(n_lights):
            bpy.ops.object.light_add(type='POINT',
                                     radius=np.random.normal(
                                         standard_height, standard_height * 0.1),
                                     align='WORLD',
                                     location=Vector(lights_center[i]),
                                     scale=(1, 1, 1))
            bpy.context.object.data.energy = np.random.normal(50, 20)
    elif use_light == "hdr":
        # load hdr env map
        hdr_file = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'irrmaps/aerodynamics_workshop_2k.hdr')
        hdr_image = bpy.data.images.load(hdr_file)

        # setup scene (world) texture
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world

        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # remove current nodes as we need to reduce effects of other output heads
        for node in nodes:
            nodes.remove(node)

        # create new nodes for env map
        environment_texture_node = nodes.new("ShaderNodeTexEnvironment")
        environment_texture_node.image = hdr_image
        output_node = nodes.new("ShaderNodeOutputWorld")

        # connect the nodes
        links.new(
            environment_texture_node.outputs["Color"], output_node.inputs["Surface"])
        bpy.context.scene.render.film_transparent = True

    else:
        print('ERROR invalid light mode ', use_light)
        raise NotImplementedError


    # 设置输出路径和文件格式
    action = largest_obj.animation_data.action
    start = int(action.frame_range[0])
    end = int(action.frame_range[1])

    bpy.context.scene.render.filepath = os.path.join(output_folder, 'frame_')
    max_frame = end-start
    bpy.context.scene.frame_end = max_frame

    #设置渲染器
    #####################################################################
    if engine_type == "cycles":
        bpy.context.scene.render.engine = 'CYCLES'
    elif engine_type == "eevee":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    else:
        print("not support engine_type")
        
    
    if device == "GPU":
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            if 'Intel' in d["name"]: 
                d["use"] = 0 
            else:
                d["use"] = 1
            print(d["name"],",", d["id"],",",d["type"],",",d["use"])

        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.color_depth = '16'
        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_normal = True
        bpy.context.scene.render.image_settings.file_format = 'PNG'
    else:
        bpy.context.scene.cycles.device = 'CPU'
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.color_depth = '32'
        bpy.context.view_layer.use_pass_z = True
        bpy.context.view_layer.use_pass_normal = True

    # world = bpy.context.scene.world
    # world.use_nodes = True

    # bg_node = world.node_tree.nodes.get("Background")

    # if bg_node:
    #     bg_node.inputs["Color"].default_value = (1,1,1,1)

    # 渲染动画
    bpy.ops.render.render(animation=True)

   