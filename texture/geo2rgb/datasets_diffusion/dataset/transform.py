import torch

def get_fru_convention(fru_convention="xyz", homogeneous=False):
    cols = []
    sign = 1
    for c in fru_convention:
        if c == "-":
            sign *= -1
            continue
        elif c == "+":
            continue
        elif c == "x" or c == "X":
            cols.append([sign, 0, 0])
            sign = 1
        elif c == "y" or c == "Y":
            cols.append([0, sign, 0])
            sign = 1
        elif c == "z" or c == "Z":
            cols.append([0, 0, sign])
            sign = 1
        else:
            raise ValueError
    
    if len(cols) != 3:
        raise ValueError
    
    transform = torch.tensor(cols, dtype=torch.float32).transpose(0,1)
    assert (transform @ torch.inverse(transform) - torch.eye(3)).abs().max() == 0
    
    if homogeneous:
        transform_ = torch.eye(4, dtype=torch.float32)
        transform_[:3,:3] = transform
        transform = transform_
    
    return transform

def get_convention(convention_str, homogeneous=False):
    if convention_str == "y-up":
        convention_str = "zxy"
    elif convention_str == "z-up":
        convention_str = "-yxz"
    return get_fru_convention(convention_str, homogeneous)

def transform_pcd(coordinate_system, obj2world, *pcds):
    sys_type, sys_convention = coordinate_system.split(",")
    sys_type = sys_type.strip()
    sys_convention = sys_convention.strip()
    
    assert sys_type in ["refcam", "object"]
    if sys_type == "refcam":
        transform, *_ = change_convention(world_fru=sys_convention)
    else:
        transform = change_convention(object_fru=sys_convention)[1] @ torch.inverse(obj2world)
        
    transform = transform / transform[...,-1:,-1:]
    
    ret_pcds = []
    for pcd in pcds:
        pcd = pcd @ transform[...,:3,:3].transpose(-1,-2) + transform[...,:3,-1:].transpose(-1,-2)
        ret_pcds.append(pcd)
    
    return ret_pcds

def change_convention(world_fru="xyz", object_fru="-yxz", normal_map_fru="zxy"):
    '''
    transform dataset's default coordinate system to target ones, returns transformation matrices
    
    our dataset defines three different coordinate systems:
    - world: our world system is always aligned to reference camera, with world's front direction always
             facing reference cam, and world's up being reference cam's up. source cameras' azimuth 
             and elevation angles are defined in this world system.
             returned xyz maps, point clouds, point normals and mesh verts are also defined in world system.
    - object: this is the system for raw mesh and raw point data that are invisible to user. our dataset performs 
              object to world transform under the hood and return everything in world system. 
              if you need object space data (e.g. point cloud) that are fixed regardless of reference camera pose, 
              use the transformation matrix "o2w" to convert from world space. e.g. 
              points_object = points_world @ torch.inverse(o2w).transpose(-1,-2)
    - normal_map: this is the system where rendered normal maps are defined in. 
    
    args:
        - item_or_batch: an item from dataset, or a collated batch
        - worldfru: a string that defines the front-right-up directions of
                    target world coordinate system, e.g. "xyz" means +x is front, +y right and +z up
                    and "-yxz" means -y is front, +x right and +z up. our dataset defaults to "xyz"
        - object_fru: target object coordinate system convention, our dataset defaults to blender's "-yxz" (aka z-up)
        - normal_map_fru: target coordinate system for rendered normal images, our dataset defaults to "zxy" (aka y-up)
        
    Returns a triplet of transformation matrices:
        - to_world: a [4,4] tensor that transforms from dataset's world (i.e. where xyz, point clouds and point normals are defined)
                    to target convention, the translational component is always zero
        - to_object: a [4,4] tensor that transforms from dataset's object to target object space convention, the translational component is always zero
        - to_normal: a [3,3] tensor that transforms from dataset's rendered normal images to target convention, the translational component is always zero
    '''
    
    to_world4 = get_convention(world_fru, homogeneous=True)
    to_object4 = get_convention(object_fru, homogeneous=True) @ torch.inverse(get_convention("-yxz", homogeneous=True))
    to_normal3 = get_convention(normal_map_fru) @ torch.inverse(get_convention("zxy"))
    
    return to_world4, to_object4, to_normal3
    
        
    