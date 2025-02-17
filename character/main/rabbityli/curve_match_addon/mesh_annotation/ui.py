import bpy


class Label3DPanel(bpy.types.Panel):
    bl_idname = "Annotation3D"
    bl_label = "Annotation3D Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Annotation3D'
    bl_context = "mesh_edit"



    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # Display the list of vertices in the group
        # layout.label(text="Vertices num:%d"%(len(obj.data.vertices)))

        # layout.separator()

        layout.label(text="Switch Curve:")
        layout.row().prop(context.window_manager.label3D_tool, "class_name")
        layout.operator("object.curve_selection", text="switch curves")


        layout.operator("object.edge_start", text="Mark as Start edge")
        layout.operator("object.edge_end", text="Mark as End edge & Save")
        # layout.operator("object.dump_curve", text="Dump all curves")

        layout.operator("object.viz_all", text="Show all curves ")
        # layout.operator("object.refresh_json", text="Refresh Json ")


class LoadMeshPanel(bpy.types.Panel):
    bl_idname = "LoadMeshNPY"
    bl_label = "LoadMeshNPY Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LoadMeshNPY'
    bl_context = "objectmode"



    def draw(self, context):
        layout = self.layout

        layout.operator("object.load_mesh", text="load mesh config json")



UI_CLASSES = [
    Label3DPanel,
    LoadMeshPanel,
    # DropDownPanel
]

