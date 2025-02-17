import taichi as ti
import numpy as np
import math
from sdf import *

@ti.data_oriented
class LevelSetSdfModel(SdfModel):
    def __init__(self, vertices, faces, scale, offset, grid_size):
        super().__init__(fixed=False)
        self.vertices = vertices
        self.faces = faces
        self.scale = scale
        self.offset = offset

        @ti.kernel
        def normalizeX(vertices: ti.template(), scale : ti.f32, offset : ti.template()):
            for I in ti.grouped(vertices):
                vertices[I] = vertices[I] * scale + offset

        normalizeX(self.vertices, scale, offset)
        self.indices = ti.field(ti.i32, shape=self.faces.shape[0])

        self.size = grid_size
        self.dx = 1 / grid_size
        self.phi_closest_tri = ti.field(ti.i64, shape=(grid_size, grid_size, grid_size))
        self.sqrt_3 = math.sqrt(3.0)
        self.intersection_count = ti.field(ti.i32, shape=(grid_size, grid_size, grid_size))

        self.exact_band = 1

        self.initIndices()
        self.makeLevelSet()

    @ti.kernel
    def calc_particles_sdf_and_normal(self, particles: ti.template(), sdf: ti.template(), sdf_grad: ti.template()):
        for I in ti.grouped(particles):
            sdf[I] = self.dist(particles[I] * self.scale)
            # sdf[I] = 1.0
            # sdf[I] = 0.0
            sdf_grad[I] = self.normal(particles[I] * self.scale).normalized(1e-6)

    @ti.kernel
    def calc_particles_sdf_and_normal_no_scale(self, particles: ti.template(), sdf: ti.template(), sdf_grad: ti.template()):
        for I in ti.grouped(particles):
            sdf[I] = self.dist(particles[I])
            # sdf[I] = 1.0
            # sdf[I] = 0.0
            sdf_grad[I] = self.normal(particles[I]).normalized(1e-6)
    
    def update(self, t):
        pass

    @ti.func
    def dist(self, pos):
        I = ti.cast(pos / self.dx - 0.5, int)
        phi, _id = self.decode(self.phi_closest_tri[I])
        return phi
    
    @ti.func
    def normal(self, pos):
        I = ti.cast(pos / self.dx - 0.5, int)
        n = ti.Vector.zero(ti.f32, 3)
        phi_0, _0 = self.decode(self.phi_closest_tri[I])
        for x in ti.static(range(3)):
            phi_1, _1 = self.decode(self.phi_closest_tri[I + ti.Vector.unit(3, x)])
            n[x] = phi_1 - phi_0
        return n.normalized(1e-12)

    def render(self, scene):
        scene.mesh(self.vertices, self.indices, color = (0.7, 0.3, 0.3), two_sided = True)

    @ti.kernel
    def initIndices(self):
        for I in ti.grouped(self.faces):
            self.indices[I] = self.faces[I]

    @ti.func
    def quantized(self, x):
        return ti.cast(x / self.sqrt_3 * 4294967296.0, ti.i64)

    @ti.func
    def encode(self, dist, id):
        return (self.quantized(dist) << 30) | id

    @ti.func
    def decode(self, val):
        dist = (val >> 30) / 4294967296.0
        id = ti.cast(val & ((ti.cast(1, ti.i64) << 30) - 1), ti.i32)
        return dist, id

    @ti.func
    def checkNeighbour(self, I0, I):
        _d0, id0 = self.decode(self.phi_closest_tri[I0])
        if id0 < ((1 << 30) - 1):
            p, q, r = self.indices[id0*3+0], self.indices[id0*3+1], self.indices[id0*3+2]
            d = self.pointTriangleDistance((I + 0.5) * self.dx, 
                                            self.vertices[p],
                                            self.vertices[q],
                                            self.vertices[r])
            d1, _id = self.decode(self.phi_closest_tri[I])
            if d < d1:
                self.phi_closest_tri[I] = self.encode(d, id0)

    @ti.kernel
    def sweep(self):
        for i, j in ti.ndrange(self.size, self.size):
            k = 0
            while k < self.size-1:
                self.checkNeighbour(ti.Vector([i,j,k]), ti.Vector([i,j,k+1]))
                k += 1
        for i, j in ti.ndrange(self.size, self.size):
            k = self.size-1
            while k >= 1:
                self.checkNeighbour(ti.Vector([i,j,k]), ti.Vector([i,j,k-1]))
                k -= 1
        for i, j in ti.ndrange(self.size, self.size):
            k = 0
            while k < self.size-1:
                self.checkNeighbour(ti.Vector([k,i,j]), ti.Vector([k+1,i,j]))
                k += 1
        for i, j in ti.ndrange(self.size, self.size):
            k = self.size-1
            while k >= 1:
                self.checkNeighbour(ti.Vector([k,i,j]), ti.Vector([k-1,i,j]))
                k -= 1
        for i, j in ti.ndrange(self.size, self.size):
            k = 0
            while k < self.size-1:
                self.checkNeighbour(ti.Vector([i,k,j]), ti.Vector([i,k+1,j]))
                k += 1
        for i, j in ti.ndrange(self.size, self.size):
            k = self.size-1
            while k >= 1:
                self.checkNeighbour(ti.Vector([i,k,j]), ti.Vector([i,k-1,j]))
                k -= 1

    def makeLevelSet(self):
        @ti.kernel
        def initPhi(phi : ti.template()):
            for I in ti.grouped(phi):
                phi[I] = self.quantized(self.sqrt_3) << 30 | ((1 << 30) - 1)

        initPhi(self.phi_closest_tri)
        self.intersection_count.fill(0) # intersection_count(i,j,k) is # of tri intersections in (i-1,i]x{j}x{k}
        # we begin by initializing distances near the mesh, and figuring out intersection counts
        self.initializeDistance()

        # and now we fill in the rest of the distances with fast sweeping
        for passes in range(4):
            self.sweep()

        # then figure out signs (inside/outside) from intersection counts
        self.calcSign()

    @ti.kernel
    def calcSign(self):
        for j, k in ti.ndrange(self.size, self.size):
            total_count = 0
            for i in range(self.size):
                total_count += self.intersection_count[i, j, k]
                if total_count % 2 == 1: # if parity of intersections so far is odd,
                    phi, id = self.decode(self.phi_closest_tri[i, j, k])
                    self.phi_closest_tri[i, j, k] = self.encode(-phi, id)

    # find distance x0 is from segment x1-x2
    @ti.func
    def pointSegmentDistance(self, x0, x1, x2):
        dx = x2 - x1
        m2 = dx.norm_sqr()
        # find parameter value of closest point on segment
        s12=(float)(x2-x0).dot(dx)/m2
        if s12<0.0:
            s12=0.0
        if s12>1.0:
            s12=1.0
        
        # and find the distance
        return (x0 - (s12*x1+(1-s12)*x2)).norm()

    # find distance x0 is from triangle x1-x2-x3
    @ti.func
    def pointTriangleDistance(self, x0, x1, x2, x3):
        # first find barycentric coordinates of closest point on infinite plane
        x13 = x1 - x3
        x23 = x2 - x3
        x03 = x0 - x3
        m13 = x13.norm_sqr()
        m23 = x23.norm_sqr()
        d = x13.dot(x23)
        invdet=1. / max(m13*m23-d*d, 1e-30)
        a = x13.dot(x03)
        b = x23.dot(x03)
        # the barycentric coordinates themselves
        w23=invdet*(m23*a-d*b)
        w31=invdet*(m13*b-d*a)
        w12=1.0-w23-w31

        result=0.0
        if w23>=0 and w31>=0 and w12>=0:
            result = (x0 - (w23*x1+w31*x2+w12*x3)).norm()
        else: # we have to clamp to one of the edges
            if w23>0: # this rules out edge 2-3 for us
                result = min(self.pointSegmentDistance(x0,x1,x2), self.pointSegmentDistance(x0,x1,x3))
            elif w31>0: # this rules out edge 1-3
                result = min(self.pointSegmentDistance(x0,x1,x2), self.pointSegmentDistance(x0,x2,x3))
            else: # w12 must be >0, ruling out edge 1-2
                result = min(self.pointSegmentDistance(x0,x1,x3), self.pointSegmentDistance(x0,x2,x3))
        
        return result
    
    # calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
    # return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate triangle)
    @ti.func
    def orientation(self, x1, y1, x2, y2):
        twice_signed_area = y1*x2-x1*y2
        sign = 0
        if twice_signed_area > 0: sign = 1
        elif twice_signed_area < 0: sign = -1
        elif y2>y1: sign = 1
        elif y2<y1: sign = -1
        elif x1>x2: sign = 1
        elif x1<x2: sign = -1
        else: sign = 0 # only true when x1==x2 and y1==y2
        return sign, twice_signed_area

    # robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
    # if true is returned, the barycentric coordinates are set in a,b,c.
    @ti.func
    def pointInTriangle2d(self, x0, y0, x1, y1, x2, y2, x3, y3):
        inside = True
        a, b, c = 0.0, 0.0, 0.0
        x1 -= x0
        x2 -= x0
        x3 -= x0
        y1 -= y0
        y2 -= y0
        y3 -= y0
        signa, a = self.orientation(x2, y2, x3, y3)
        if signa == 0: inside = False
        signb, b = self.orientation(x3, y3, x1, y1)
        if signb != signa: inside = False
        signc, c = self.orientation(x1, y1, x2, y2)
        if signc != signa: inside = False
        sum = a + b + c
        if sum != 0.0: # if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
            a /= sum
            b /= sum
            c /= sum
        
        return inside, a, b, c

    @ti.kernel
    def initializeDistance(self):
        for I in range(self.faces.shape[0]//3):
            p, q, r = self.vertices[self.faces[I*3]], self.vertices[self.faces[I*3+1]], self.vertices[self.faces[I*3+2]]
            # coordinates in grid to high precision
            fip = p.x / self.dx - 0.5
            fjp = p.y / self.dx - 0.5
            fkp = p.z / self.dx - 0.5
            fiq = q.x / self.dx - 0.5
            fjq = q.y / self.dx - 0.5
            fkq = q.z / self.dx - 0.5
            fir = r.x / self.dx - 0.5
            fjr = r.y / self.dx - 0.5
            fkr = r.z / self.dx - 0.5
            # do distances nearby
            i0 = clamp(int(min(fip,fiq,fir))-self.exact_band, 0, self.size-1)
            i1 = clamp(int(max(fip,fiq,fir))+self.exact_band+1, 0, self.size-1)
            j0 = clamp(int(min(fjp,fjq,fjr))-self.exact_band, 0, self.size-1)
            j1 = clamp(int(max(fjp,fjq,fjr))+self.exact_band+1, 0, self.size-1)
            k0 = clamp(int(min(fkp,fkq,fkr))-self.exact_band, 0, self.size-1)
            k1 = clamp(int(max(fkp,fkq,fkr))+self.exact_band+1, 0, self.size-1)
            for k in range(k0, k1+1):
                for j in range(j0, j1+1):
                    for i in range(i0, i1+1):
                        gx = ti.Vector([(i+0.5)*self.dx, (j+0.5)*self.dx, (k+0.5)*self.dx])
                        d = self.pointTriangleDistance(gx, p, q, r)
                        ti.atomic_min(self.phi_closest_tri[i, j, k], self.encode(d, I)) # use I.x to replace t.id ???
                        # if d < self.phi[i, j, k]:
                        #     self.phi[i, j, k] = d
                        #     self.closest_tri[i, j, k] = t.id

            # and do intersection counts
            j0=clamp(ti.cast(ti.ceil(min(fjp,fjq,fjr)), int), 0, self.size-1)
            j1=clamp(ti.cast(ti.floor(max(fjp,fjq,fjr)), int), 0, self.size-1)
            k0=clamp(ti.cast(ti.ceil(min(fkp,fkq,fkr)), int), 0, self.size-1)
            k1=clamp(ti.cast(ti.floor(max(fkp,fkq,fkr)), int), 0, self.size-1)
            for k in range(k0, k1+1):
                for j in range(j0, j1+1):
                    inside, a, b, c = self.pointInTriangle2d(float(j), float(k), fjp, fkp, fjq, fkq, fjr, fkr)
                    if inside:
                        fi=a*fip+b*fiq+c*fir # intersection i coordinate
                        i_interval=int(ti.ceil(fi)) # intersection is in (i_interval-1,i_interval]
                        if i_interval < 0: 
                            self.intersection_count[0, j, k] += 1 # we enlarge the first interval to include everything to the -x direction
                        elif i_interval < self.size:
                            self.intersection_count[i_interval, j, k] += 1
                        # we ignore intersections that are beyond the +x side of the grid

# @ti.data_oriented
# class MotionLevelSetSdfModel(LevelSetSdfModel):
#     def __init__(self, frame_dt, filename, dict_name, max_frame, scale, offset, grid_size):
#         super().__init__(filename, scale, offset, grid_size)
#         self.scale = scale
#         self.offset = offset
#         self.frame_dt = frame_dt
#         self.max_frame = max_frame
#         self.dict_name = dict_name
#         self.current_frame = 0

#         self.end_motion = self.load(1)
#         self.model.verts.x.from_numpy(self.end_motion)
#         self.next_phi_closest_tri = ti.field(ti.i64, shape=(grid_size, grid_size, grid_size))
#         self.makeLevelSet()
#         self.next_phi_closest_tri.copy_from(self.phi_closest_tri)

#         self.start_motion = self.load(0)
#         self.model.verts.x.from_numpy(self.start_motion)
#         self.makeLevelSet()

#         self.interval = ti.field(ti.f32, shape=())

    
#     def update(self, t):
#         if t >= self.frame_dt * self.max_frame:
#             return
#         if t >= (self.current_frame + 1) * self.frame_dt:
#             self.current_frame += 1
#             self.start_motion = self.end_motion
#             self.end_motion = self.load(self.current_frame + 1)
#             self.model.verts.x.from_numpy(self.end_motion)
#             self.makeLevelSet()
#             self.next_phi_closest_tri.copy_from(self.phi_closest_tri)
#             self.model.verts.x.from_numpy(self.start_motion)
#             self.makeLevelSet()

#         self.interval[None] = t / self.frame_dt - self.current_frame
#         assert self.interval[None] >=0 and self.interval[None] <= 1
#         print("upd: ", self.interval[None])

#     def load(self, frame):
#         return np.load(f"{self.dict_name}/{frame+1}.npy") * self.scale + self.offset
    
#     @ti.func
#     def phi(self, I):
#         phi_0, _id0 = self.decode(self.phi_closest_tri[I])
#         phi_1, _id1 = self.decode(self.next_phi_closest_tri[I])
#         return phi_1 * self.interval[None] + phi_0 * (1 - self.interval[None])

#     @ti.func
#     def dist(self, pos):
#         I = ti.cast(pos / self.dx - 0.5, int)
#         return self.phi(I)
    
#     @ti.func
#     def normal(self, pos):
#         I = ti.cast(pos / self.dx - 0.5, int)
#         n = ti.Vector.zero(ti.f32, 3)
#         phi_0 = self.phi(I)
#         for x in ti.static(range(3)):
#             phi_1 = self.phi(I + ti.Vector.unit(3, x))
#             n[x] = phi_1 - phi_0
#         return n.normalized(1e-12)
    
#     @ti.func
#     def check(self, pos, vel):
#         phi = self.dist(pos)
#         inside = False
#         dotnv = 0.0
#         diff_vel = ti.Vector.zero(ti.f32, 3) # FIXME: diff_vel cannot calc from phi
#         n = ti.Vector.zero(ti.f32, 3)
#         if phi < 0.0:
#             n = self.normal(pos)
#             I = ti.cast(pos / self.dx - 0.5, int)
#             phi0, _0 = self.decode(self.phi_closest_tri[I])
#             phi1, _1 = self.decode(self.next_phi_closest_tri[I])
#             solid_vel_dot_n = (phi0 - phi1) / self.frame_dt
#             if solid_vel_dot_n < 0.0:
#                 solid_vel_dot_n = 0.0
#             else:
#                 solid_vel_dot_n *= 30.0
#             dotnv = solid_vel_dot_n - n.dot(vel)
#             if dotnv > 0.0:
#                 inside = True
        
#         return self.fixed[None], inside, dotnv, diff_vel, n
