import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator
import os
import bmesh
import  numpy as np
from bmesh.types import BMEdge
import pathlib
from pathlib import  Path
import os,  sys


from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, OperatorFileListElement
from bpy.props import CollectionProperty, StringProperty

sys.path.append ( os.path.join(os.path.dirname(__file__)) )

from utils import sort_edge, load_json, write_json, select_from_lst, unselect_all

# codedir = os.path.dirname(os.path.abspath(__file__))
# config_json = os.path.join( codedir, "config.json")
# config_json = "/home/rabbityl/Desktop/curve_match_addon/mesh_annotation/config.json"



# jdata = load_json( config_json )
# curve_names = jdata ["curve_names"]
# mesh_path = jdata["mesh_npy_path"]
# dump_path = os.path.join( pathlib.Path(mesh_path).parent, "curve.json")
#
#
#
# curves = {}
# for name in curve_names:
#     curves[name] = {"edge": [], "edge_index": []}

curve_names,  dump_path, mesh_path, curve_names, curves, config_json = None,None,None,None,None,None

def refresh():
    global curve_names,  dump_path, mesh_path, curve_names, curves, config_json
    jdata = load_json( config_json )
    curve_names = jdata["curve_names"]
    mesh_path = jdata["mesh_npy_path"]
    dump_path = os.path.join(pathlib.Path(mesh_path).parent, "curve.json")



    # reuse previous data
    if os.path.exists(dump_path):
        print("curve data found, reuse")
        dump_data = load_json(dump_path)
        for name in curve_names:
            if name in dump_data:
                curves[name] = dump_data[name]



# refresh()


current_start_id = None
current_end_id = None




def dump_curves(context):
    global current_start_id, current_end_id, curves, dump_path
    print(" current_start, current_end", current_start_id, current_end_id)
    print("inside DumpCurve()")
    me = context.object.data
    bm = bmesh.from_edit_mesh(me)

    # Get the selected edge
    selected_edges = [e for e in bm.edges if e.select]

    assert current_start_id is not None
    assert current_end_id is not None

    contain_start = False
    contain_end = False

    edge_np = []

    edge_index = []

    for idx, e in enumerate(selected_edges):
        print(idx, "/", len(selected_edges), e)
        edge_index.append(e.index)
        if e.index == current_start_id:
            contain_start = True
            print("start idx", idx)
            s_idx = [[e.verts[0].index, e.verts[1].index]]
        elif e.index == current_end_id:
            contain_end = True
            print("end idx", idx)
            e_idx = [[e.verts[0].index, e.verts[1].index]]
        else:
            edge_np.append([e.verts[0].index, e.verts[1].index])

    if not (contain_start and contain_end):
        print("contain_start and contain_end", contain_start and contain_end)

    edges = s_idx + edge_np + e_idx

    print(edges)

    edges = sort_edge(edges)
    vg_name = context.window_manager.label3D_tool.class_name
    curves[vg_name]["edge"] = edges
    curves[vg_name]["edge_index"] = edge_index
    print(curves)

    write_json(dump_path, curves)



class LoadMesh(bpy.types.Operator, ImportHelper):
    bl_idname = "object.load_mesh"
    bl_label = "load_mesh"

    filter_glob: StringProperty(
        default='*.json',
        options={'HIDDEN'}
    )

    some_boolean: BoolProperty(
        name='Do a thing',
        description='Do a thing with the file you\'ve selected',
        default=True,
    )

    def execute(self, context):  # execute() is called when running the operator.

        global  curve_names,  dump_path, mesh_path, curve_names, curves, config_json

        if curve_names is not None: # second time running this
            from . import properties

            properties.destroy_props()
            for prop_class in reversed(properties.PROPERTY_CLASSES):
                bpy.utils.unregister_class(prop_class)


        curve_names, dump_path, mesh_path, curve_names, curves, config_json = None, None, None, None, None, None


        _, extension = os.path.splitext(self.filepath)

        config_json = self.filepath

        print('Selected file:', self.filepath)
        # print('File name:', filename)
        # print('File extension:', extension)
        # print('Some Boolean:', self.some_boolean)

        #####
        jdata = load_json(config_json)
        curve_names = jdata["curve_names"]
        mesh_path = jdata["mesh_npy_path"]
        dump_path = os.path.join(pathlib.Path(mesh_path).parent, "curve.json")

        curves = {}
        for name in curve_names:
            curves[name] = {"edge": [], "edge_index": []}

        print( "curve_names", curve_names)
        #####

        # reuse previous data
        if os.path.exists(dump_path):
            print("curve data found, reuse")
            dump_data = load_json(dump_path)
            for name in curve_names:
                if name in dump_data:
                    curves[name] = dump_data[name]

        # mesh_ext =  mesh_path.split("/")[-1].split(".")[-1]
        npy = mesh_path
        with open(npy, 'rb') as f:
            vert_data = np.load(f)
            face_data = np.load(f)



        # make object mesh
        vertices = vert_data.tolist()
        edges = []
        faces = face_data.tolist()
        mesh_data = bpy.data.meshes.new('mesh_data')
        mesh_data.from_pydata(vertices, edges, faces)
        mesh_data.update()
        the_mesh = bpy.data.objects.new('the_mesh', mesh_data)
        the_mesh.data.vertex_colors.new()  # init color
        bpy.context.collection.objects.link(the_mesh)





        # add properties dynamically
        from . import properties
        for prop_class in properties.PROPERTY_CLASSES:
            bpy.utils.register_class(prop_class)
        properties.define_props()


        return {'FINISHED'}



class MarkStart(bpy.types.Operator):
    bl_idname = "object.edge_start"
    bl_label = "note_start_edge"
    def execute(self, context):

        global  current_start_id
        print( "current_start", current_start_id)

        me = context.object.data
        bm = bmesh.from_edit_mesh(me)
        start = bm.select_history[-1]
        assert isinstance (bm.select_history[-1], BMEdge)
        current_start_id = start.index
        print(current_start_id, "current_start")
        return {'FINISHED'}

class MarkEnd(bpy.types.Operator):
    bl_idname = "object.edge_end"
    bl_label = "note_end_edge"

    def execute(self, context):

        global current_end_id
        print( "current_end", current_end_id)

        me = context.object.data
        bm = bmesh.from_edit_mesh(me)
        end = bm.select_history[-1]
        assert isinstance(bm.select_history[-1], BMEdge)
        current_end_id = end.index
        print(current_end_id, "current_end")


        dump_curves( context )

        return {'FINISHED'}



class DumpCurve(bpy.types.Operator):
    bl_idname = "object.dump_curve"
    bl_label = "dump_curve"

    def execute(self, context):

        global current_start_id, current_end_id, curves, dump_path
        print( " current_start, current_end", current_start_id, current_end_id)
        print( "inside DumpCurve()")
        me = context.object.data
        bm = bmesh.from_edit_mesh(me)

        # Get the selected edge
        selected_edges = [e for e in bm.edges if e.select]

        assert current_start_id is not None
        assert current_end_id is not None

        contain_start = False
        contain_end = False


        edge_np = []

        edge_index = []


        for idx, e in  enumerate (selected_edges):
            print( idx,"/", len(selected_edges), e )
            edge_index.append(e.index)
            if e.index == current_start_id:
                contain_start = True
                print("start idx", idx)
                s_idx = [[ e.verts[0].index, e.verts[1].index ]]
            elif e.index == current_end_id:
                contain_end = True
                print("end idx", idx)
                e_idx = [[ e.verts[0].index, e.verts[1].index ]]
            else:
                edge_np.append([e.verts[0].index, e.verts[1].index])


        if not (contain_start and contain_end):
            print( "contain_start and contain_end", contain_start and contain_end )

        edges = s_idx + edge_np + e_idx


        print( edges )


        edges = sort_edge(edges)
        vg_name = context.window_manager.label3D_tool.class_name
        curves [vg_name]["edge"] = edges
        curves [vg_name]["edge_index"] = edge_index
        print( curves )


        write_json( dump_path, curves)

        # current_start_id = None
        # current_end_id = None

        # visualize all available curves
        unselect_all(context)
        all_curves = []
        for k in curves.keys():
            all_curves = all_curves + curves [k]["edge_index"]
        select_from_lst(context, all_curves)
        print(vg_name)


        return {'FINISHED'}



class VizAll(bpy.types.Operator):
    bl_idname = "object.viz_all"
    bl_label = "viz_all"

    def execute(self, context):

        global current_start_id, current_end_id, curves, dump_path

        unselect_all(context)
        all_curves = []
        for k in curves.keys():
            all_curves = all_curves + curves [k]["edge_index"]
        select_from_lst(context, all_curves)
        # print(vg_name)


        return {'FINISHED'}



class CurveSwitch(bpy.types.Operator):
    bl_idname = "object.curve_selection"
    bl_label = "Switch curve"


    def execute(self, context):

        global curves

        bpy.ops.mesh.select_mode(type="EDGE")
        unselect_all(context)
        vg_name = context.window_manager.label3D_tool.class_name
        select_from_lst(context, curves[vg_name]["edge_index"] )
        print(vg_name)
        return {'FINISHED'}


# class RefreshJson(bpy.types.Operator):
#     bl_idname = "object.refresh_json"
#     bl_label = "refresh_json"
#
#
#     def execute(self, context):
#         refresh()
#         return {'FINISHED'}



OPERATORS = [
    MarkStart,
    MarkEnd,
    # DumpCurve,
    CurveSwitch,
    LoadMesh,
    VizAll,
    # RefreshJson
    # DropDownExample
]

