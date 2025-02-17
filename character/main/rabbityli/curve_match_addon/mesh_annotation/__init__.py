bl_info = {
    "name": "annotate mesh curves in 3D",
    "author": "rabbityli in tencent",
    "version": (2024, 1.0),
    "blender": (2, 92, 0),
    "location": "Viewport > Right panel",
    "description": "label mesh curves",
    "category": "annotate_curves",
}


def register():
    import bpy
    from . import (
        operation,
        ui,
        # properties,
    )
    # for prop_class in properties.PROPERTY_CLASSES:
    #     bpy.utils.register_class(prop_class)

    for ui_class in ui.UI_CLASSES:
        bpy.utils.register_class(ui_class)

    for operator in operation.OPERATORS:
        bpy.utils.register_class(operator)
    
    # properties.define_props()




def unregister():
    import bpy
    from . import (
        operation,
        ui,
        properties,
    )
    # properties.destroy_props()

    for operator in reversed(operation.OPERATORS):
        bpy.utils.unregister_class(operator)

    for ui_class in reversed(ui.UI_CLASSES):
        bpy.utils.unregister_class(ui_class)

    # for prop_class in reversed(properties.PROPERTY_CLASSES):
    #     bpy.utils.unregister_class(prop_class)
