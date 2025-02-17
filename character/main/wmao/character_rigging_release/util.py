import numpy as np

def axis_angle_to_rot(axis, angle):
    """
    Convert axis angle representation to rotation matrix
    :param axis: Axis of rotation within ["X", "Y", "Z"]
    :param angle: Angle of rotation in radians
    :return: Rotation matrix"""
    cos = np.cos(angle)
    sin = np.sin(angle)
    one = 1
    zero = 0

    if axis == "X":
        R_flat = np.array([one, zero, zero, zero, cos, -sin, zero, sin, cos])
    elif axis == "Y":
        R_flat = np.array([cos, zero, sin, zero, one, zero, -sin, zero, cos])
    elif axis == "Z":
        R_flat = np.array([cos, -sin, zero, sin, cos, zero, zero, zero, one])
    else:
        raise ValueError("letter must be either X, Y or Z.")
    return np.reshape(R_flat, (3, 3))
