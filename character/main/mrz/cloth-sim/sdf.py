import taichi as ti
from sdf_utils.utils import *

@ti.data_oriented
class SdfModel:
    def __init__(self, fixed):
        self.vel = ti.Vector.field(3, float, shape=())
        self.fixed = ti.field(int, shape=())
        self.fixed[None] = fixed

    @ti.func
    def check(self, pos, vel):
        phi = self.dist(pos)
        inside = False
        dotnv = 0.0
        diff_vel = ti.Vector.zero(ti.f32, 3)
        n = ti.Vector.zero(ti.f32, 3)
        if phi < 0.0:
            n = self.normal(pos)
            diff_vel = self.vel[None] - vel
            dotnv = n.dot(diff_vel)
            if dotnv > 0.0 or self.fixed[None]:
                inside = True
        
        return self.fixed[None], inside, dotnv, diff_vel, n

@ti.data_oriented
class NullSdfModel(SdfModel):
    def __init__(self):
        super().__init__(fixed=False)

    def update(self, t):
        pass

    def render(self, scene):
        pass

    @ti.func
    def dist(self, pos):
        return 1.0
    
    @ti.func
    def normal(self, pos):
        return ti.Vector.zero(ti.f32, 3)

@ti.data_oriented
class SphereSdfModel(SdfModel):
    def __init__(self,
        height = 0.3,
        middle_y = 0.3,
        move_y = 0.1,
        duration = 0.1,
        radius = 0.1):
        super().__init__(fixed=False)
        self.sphere_pos = ti.Vector.field(3, float, shape=1)
        self.height = height
        self.middle_y = middle_y
        self.move_y = move_y
        self.duration = duration

        self.sphere_pos[0] = ti.Vector([0.5, self.middle_y, self.height])
        self.sphere_radius = radius

    @ti.kernel
    def update(self, t : ti.f32):
        co = PI * 2.0 / self.duration
        self.vel[None] = ti.Vector([0, co * ti.cos(t * co) * self.move_y, 0])
        self.sphere_pos[0] = ti.Vector([0.5, self.middle_y + ti.sin(t * co) * self.move_y, 
                                        self.height])

    @ti.func
    def dist(self, pos): # Function computing the signed distance field
        return (pos - self.sphere_pos[0]).norm(1e-10) - self.sphere_radius

    @ti.func
    def normal(self, pos): # Function computing the gradient of signed distance field
        return (pos - self.sphere_pos[0]).normalized(1e-10)

    def render(self, scene):
        scene.particles(self.sphere_pos, self.sphere_radius - 0.02, color = (0, 0, 1))

@ti.data_oriented
class HangSdfModel(SdfModel):
    def __init__(self, pos, release_time = -1.0):
        super().__init__(fixed=True)
        self.sphere_pos = ti.Vector.field(3, float, shape=pos.shape[0])
        self.sphere_pos.from_numpy(pos)
        self.sphere_radius = 0.013
        self.release_time = release_time

        self.t = ti.field(ti.f32, shape=())

    @ti.kernel
    def update(self, t : ti.f32):
        self.t[None] = t

    @ti.func
    def dist(self, pos): # Function computing the signed distance field
        dist = 1e5
        for i in range(self.sphere_pos.shape[0]):
            dist = min((pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius, dist)
        if self.release_time > 0 and self.t[None] > self.release_time:
            dist = 1.0
        return dist

    @ti.func
    def normal(self, pos): # Function computing the gradient of signed distance field
        dist = 1e5
        normal = ti.Vector.zero(ti.f32, 3)
        for i in range(self.sphere_pos.shape[0]):
            dist0 = (pos - self.sphere_pos[i]).norm(1e-9) - self.sphere_radius
            if dist0 < dist:
                dist = dist0
                normal = (pos - self.sphere_pos[0]).normalized(1e-9)
        return normal
    
    def render(self, scene):
        if self.release_time > 0 and self.t[None] > self.release_time:
            pass
        else:
            scene.particles(self.sphere_pos, self.sphere_radius, color = (1, 0, 0))

@ti.data_oriented
class MixedSdfModel(SdfModel):
    def __init__(self, sdf_a, sdf_b):
        super().__init__(fixed=False)
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b
    
    def update(self, t):
        self.sdf_a.update(t)
        self.sdf_b.update(t)
    
    def render(self, scene):
        self.sdf_a.render(scene)
        self.sdf_b.render(scene)

    @ti.func
    def dist(self, pos):
        phi_a = self.sdf_a.dist(pos)
        phi_b = self.sdf_b.dist(pos)
        return phi_a if phi_a < phi_b else phi_b
    
    @ti.func
    def check(self, pos, vel):
        phi_a = self.sdf_a.dist(pos)
        phi_b = self.sdf_b.dist(pos)

        inside = False
        dotnv = 0.0
        diff_vel = ti.Vector.zero(ti.f32, 3)
        n = ti.Vector.zero(ti.f32, 3)
        fixed = False
        fixed_0 = False
        fixed_1 = False

        if phi_a < phi_b:
            fixed_0, inside, dotnv, diff_vel, n = self.sdf_a.check(pos, vel)
        else:
            fixed_1, inside, dotnv, diff_vel, n = self.sdf_b.check(pos, vel)

        if phi_a < 0.0 and fixed_0: fixed = True
        if phi_b < 0.0 and fixed_1: fixed = True
        
        return fixed, inside, dotnv, diff_vel, n
