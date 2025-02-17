import bpy
import os
from pathlib import Path
import sys
sys.path.insert(0, "..")
from utils import *
import shutil

from mathutils import Matrix, Vector, Quaternion, Euler
def change_mat():
    for mat in bpy.data.materials:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                principled.inputs['Metallic'].default_value = 0.2
                principled.inputs['Specular'].default_value = 0.2
                principled.inputs['Roughness'].default_value = 0.8

def screen_shot(obj_center, length ,filepath):
    bpy.context.scene.render.engine = 'CYCLES'

    camera_data = bpy.data.cameras.new(name="screenshot_cam")
    camera_object = bpy.data.objects.new("screenshot_cam", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    # 设置Camera，在物体为中心的球面上采样相机
    pos = obj_center  # scale
    pos[1] = -length*2.5
    camera_object.location = Vector(pos)
    camera_object.rotation_euler[0] = 1.5708
    camera_object.rotation_euler[1] = 0
    camera_object.rotation_euler[2] = 0
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100
    bpy.context.view_layer.update()  # update camera params
    bpy.context.scene.render.resolution_x = 196
    bpy.context.scene.render.resolution_y = 196

    # change_mat()
    bpy.context.scene.camera = bpy.data.objects['screenshot_cam']
    bpy.context.scene.render.filepath = filepath  # os.path.join( dump_path, cam + ".exr")
    bpy.ops.render.render(write_still=True)

def compute_mesh_height( meshes ,skip=1) :

    #计算角色中心点,身体长度
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    verts = []
    for ind, mesh in enumerate(meshes):
        print(ind)
        vert, _ = read_mesh_to_ndarray( mesh, mode="edit", skip=skip)
        mat = np.asarray(mesh.matrix_world)
        R,t = mat[:3,:3], mat[:3,3:] #Apply World Scale
        verts.append( ( R@vert.T + t ).T )
    verts=np.concatenate(verts, axis=0)
    print("verts", verts.shape, verts.min(axis=0), verts.max(axis=0))
    min_ = verts.min(axis=0)
    max_ = verts.max(axis=0)
    obj_center = (min_ + max_) / 2
    length = max_[2] - min_[2]
    print( "edit mode ", obj_center, length)

    return obj_center, length


def save_images_smart (out_path):
    # rename_bpy_images()
    for i in range (len(bpy.data.images) ):
        image = bpy.data.images[i] #.name

        # print("----------------------------------------")
        # print( "image.name", image.name)
        # print( "image.filepath", image.filepath)

        # im_name = image.name
        im_name = image.filepath.split("/")[-1]
        # if len(im_name) > 4 and im_name[-4:] in [ ".png", ".jpg", ".tif"]:
        #     im_name = im_name [:-4]
        filepath = os.path.join(out_path, im_name )
        print( "filepath", filepath)


        # print( "os.path.isfile(image.filepath)", os.path.isfile(image.filepath))
        # shutil.copy(image.filepath, filepath)

        if os.path.isfile(image.filepath):
            shutil.copy(image.filepath, filepath)

        else :
            try :
                image.save_render( filepath )
            except Exception as e:
                print( str(e) )

        # print("----------------------------------------")



def run( character_fbx , output_path, obj_name, apply_scale=False  ):

    tex_path = os.path.join(output_path, "textures")

    p = Path(output_path)
    p.mkdir(parents=True, exist_ok=True)
    p = Path(tex_path)
    p.mkdir(parents=True, exist_ok=True)

    #remove all default objects in a collection
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()




    # 载入fbx model
    bpy.ops.import_scene.fbx(
        filepath=character_fbx,
        use_anim=True)
    try:
        to_rest_pose()
    except:
        print ( "to_rest_pose_failure" )

    print(" import finish")


    print( "tex_path:", tex_path)

    # if uvmap_path :
    #     character_fbx_path = Path(character_fbx)
    #     fbx_parent = str(character_fbx_path.parent)
    #     src_dir = os.path.join( fbx_parent, uvmap_path )
    #     tgt_dir = tex_path
    #
    #     files = glob.iglob(os.path.join(src_dir, "*.png"))
    #
    #
    #     for file in files:
    #         if os.path.isfile(file):
    #             im_name= file.split("/")[-1]
    #             tgt_file = os.path.join( tex_path, im_name )
    #             print( "file, tgt_file", file, tgt_file)
    #             try:
    #                 shutil.copy(file, tgt_file)
    #             except:
    #                 pass
        #
        # print("src_dir:", src_dir)
        # print("tgt_dir:", tgt_dir)
        # files = os.listdir(src_dir)
        # shutil.copytree(src_dir, tgt_dir)
        # exit (0)


    # else :

    save_images_smart(tex_path)



    meshes = []

    weapons = []


    for ind, obj in enumerate(bpy.data.objects):
        if obj.type == 'MESH':

            print( obj.name)

            if "weapon" in obj.name or "Weapon" in obj.name:
                weapons.append( obj )

            else:
                meshes.append(obj)


    print( meshes)


    if len(weapons) > 0 :
        print( type(weapons))
        weapons = join_list_of_mesh(weapons)
        print( type(weapons))
        # remove all default objects in a collection
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = weapons
        weapons.select_set(True)
        bpy.ops.object.delete()



    # process cloth meshes
    if len(meshes) > 0 :
        the_mesh = join_list_of_mesh( meshes )
        the_mesh = triangulate(the_mesh)


        #bake transformation
        # bpy.context.view_layer.objects.active = the_mesh
        # the_mesh.select_set(True)
        # bpy.context.object.rotation_euler[1] = -1.5708


        obj_center, length = compute_mesh_height([the_mesh])
        print( "obj_center, length", obj_center, length)
        print("render screen shot")
        filepath = os.path.join(output_path, "screenshot_"+ obj_name +".png")
        screen_shot(obj_center, length, filepath)

        global_scale = 1.75/length if apply_scale else 1
        body_mesh_path = os.path.join( output_path, "for_render.obj")
        export_mesh_obj(the_mesh, body_mesh_path, global_scale=global_scale)



        fix_names(output_path)



    else :
        print("body_mesh is None")





if __name__ == '__main__':

    import sys
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    print(argv)
    character_fbx = argv [0]
    output_path = argv [1]



    apply_scale = True

    # uvmap_path = None #  "./" # if none, save from bpy.data.images


    obj_name = output_path.split("/")[-1]
    run( character_fbx , output_path, obj_name, apply_scale = apply_scale  )

    print( "finish sucess!")


