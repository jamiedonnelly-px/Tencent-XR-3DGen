import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
    CollectionProperty,
)
from bpy.types import (
    PropertyGroup,
)

from .operation import curve_names


CLASS_NAMES = [ ]
for name in curve_names:
    CLASS_NAMES.append( ( name, name, "") )



class Lable3DProp(PropertyGroup):

    class_name: bpy.props.EnumProperty(
        name="Classes",
        description="mesh class",
        items=CLASS_NAMES
    )



PROPERTY_CLASSES = [
    Lable3DProp,
]


def define_props():
    bpy.types.WindowManager.label3D_tool = PointerProperty(type=Lable3DProp)

def destroy_props():
    del bpy.types.WindowManager.label3D_tool

