# From Chang Yu

import taichi as ti

PI = 3.14159265

vec3f = ti.types.vector(3, ti.f32)
vec4f = ti.types.vector(4, ti.f32)
mat33f = ti.types.matrix(3, 3, ti.f32)

@ti.func
def mat3cols(M, i : ti.template()):
    return ti.Vector([M[0, i], M[1, i], M[2, i]])

@ti.func
def QRDecomposition(F):
    a1 = mat3cols(F, 0)
    a2 = mat3cols(F, 1)
    a3 = mat3cols(F, 2)
    e1 = a1 / a1.norm()
    r12 = e1.dot(a2)
    e2 = (a2 - r12 * e1).normalized()
    r13 = e1.dot(a3)
    r23 = e2.dot(a3)
    e3 = (a3 - r13 * e1 - r23 * e2).normalized()
    r11 = e1.dot(a1)
    r22 = e2.dot(a2)
    r33 = e3.dot(a3)
    Q = ti.Matrix.cols([e1, e2, e3])
    R = ti.Matrix([[r11, r12, r13], [0.0, r22, r23], [0.0, 0.0, r33]])
    return Q, R

@ti.func
def cot(a, b):
    return a.dot(b) / a.cross(b).norm(1e-15)

@ti.func
def clamp(x0, a, b):
    x = x0
    if x < a: x = a
    if x > b: x = b
    return x

@ti.func
def getBarycentric(x, i_low, i_high):
    s = ti.floor(x)
    i = int(x)
    f = 0.0
    if i < i_low:
        i = i_low
        f = 0.0
    elif i > i_high - 2:
        i = i_high - 2
        f = 1.0
    else:
        f = float(x - s)

    return i, f

@ti.func
def lerp(value0, value1, f):
    return (1.0 - f) * value0 + f * value1

@ti.func
def bilerp(v00, v10, v01, v11, fx, fy):
    return lerp(lerp(v00, v10, fx), \
               lerp(v01, v11, fx),  \
               fy)

@ti.func
def trilerp(v000, v100, \
            v010, v110, \
            v001, v101, \
            v011, v111, \
            fx, fy, fz):
    return lerp(bilerp(v000, v100, v010, v110, fx, fy), \
               bilerp(v001, v101, v011, v111, fx, fy),  \
               fz)
