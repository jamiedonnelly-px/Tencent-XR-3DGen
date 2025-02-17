import taichi as ti
import numpy as np
from spatial_hashing import SpatialHasher


@ti.data_oriented
class PositionBasedDynamics:
    def __init__(self,
    cloth_mesh,
    sdf,
    bending_compliance=1e-7,
    stretch_compliance=1e2,
    particle_size=0.01,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    frame_dt=1e-2,
    dt=1e-2,
    rest_iter=1000,
    XPBD=True,
    block_size=128,
    stretch_relaxation=0.3,
    bending_relaxation=0.1):

        # constants
        self.time = 0.0
        self.eps = 1e-5
        self.frame_dt = frame_dt
        self.dt = dt
        self.gravity = ti.Vector.field(3, ti.f32, shape=())
        self.gravity[None] = ti.Vector([0.0, 0.0, 0.0])

        # stiffness
        self.bending_compliance = bending_compliance
        self.stretch_compliance = stretch_compliance
        
        # pbd parameters
        self.rest_iter = rest_iter
        self.stretch_relaxation = stretch_relaxation
        self.bending_relaxation = bending_relaxation

        # profiling parameter(s)
        self.block_size = block_size

        self.mass = 1.0

        self.scale = scale
        self.offset = offset

        # Handling Self-collision with spatial hashing
        self.particle_size = particle_size
        self.upper_bound = (1, 1, 1)
        self.lower_bound = (0, 0, 0)
        self.hash_grid_cell_size = 1.25 * self.particle_size
        max_hash_grid_res = np.ceil((np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size).astype(int)
        default_hash_grid_res = np.array([150, 150, 150])
        self.hash_grid_res = np.minimum(max_hash_grid_res, default_hash_grid_res)
        self.sh = SpatialHasher(cell_size=self.hash_grid_cell_size, grid_res=self.hash_grid_res)
        print(f"Spatial Hash, grid cell size: {self.hash_grid_cell_size}, hash grid res: {self.hash_grid_res}")
        self.sh.build()

        # Xpbd parameters
        self.XPBD = XPBD

        # cloth mesh
        self.verts_arr = cloth_mesh.vertices
        self.edges_arr = cloth_mesh.edges
        self.faces_arr = cloth_mesh.faces

        num_verts = self.verts_arr.shape[0]
        num_edges = self.edges_arr.shape[0]
        num_faces = self.faces_arr.shape[0]
        self.num_verts = num_verts
        self.num_edges = num_edges
        self.num_faces = num_faces
        print(f"verts: {self.verts_arr.shape}, edges: {self.edges_arr.shape}, faces: {self.faces_arr.shape}")


        verts = ti.types.struct(
            rest_x    = ti.math.vec3, # Rest pose
            x         = ti.math.vec3, # Initial pose of at the beginning of a time step
            new_x     = ti.math.vec3, # Running pose during the constraint solving
            delta_x   = ti.math.vec3, # Collision with rigid
            v         = ti.math.vec3,
            invM      = float,
            dp        = ti.math.vec3, # pose changes
            sdf_val   = float,
            sdf_color = ti.math.vec3,
            mu_s      = float,        # static friction coefficient
            mu_k      = float,        # kinectic friction coefficient
            reordered_idx = int,
        )


        edges = ti.types.struct(
            v1                 = int,
            v2                 = int,
            rest_len           = float,
            lambda_stretch     = float,
            lambda_bending     = float,
            stretch_compliance = float,
            bending_compliance = float,
        )
        
        self.edge2faces = ti.field(int, shape=(self.num_edges, 2))
        self.faces = ti.field(int, shape=(self.num_faces, 3))
        self.rest_dihedral_angles = ti.field(float, shape=num_edges)

        self.verts = verts.field(shape=self.num_verts, layout=ti.Layout.SOA)
        self.verts_reordered = verts.field(shape=self.num_verts, layout=ti.Layout.SOA) # For spatial hashing sort use

        self.edges = edges.field(shape=self.num_edges, layout=ti.Layout.SOA)
        self.edges_reordered = edges.field(shape=self.num_edges, layout=ti.Layout.SOA)


        # Static and kinect friction ratio
        self.verts.mu_s.fill(0.25)
        self.verts.mu_k.fill(0.15)

        # collision
        self.sdf = sdf
        self.collision_iter_n = 1

        # Initialize the states
        self.reset()
    

    def reset(self):
        # Initialize with loaded data
        self.verts.x.from_numpy(self.verts_arr)
        self.edges.v1.from_numpy(self.edges_arr[:, 0])
        self.edges.v2.from_numpy(self.edges_arr[:, 1])
        self.faces.from_numpy(self.faces_arr)

        # Rest pose
        self.verts.rest_x.from_numpy(self.verts_arr)

        self.initialize()

        # Compute edge to face relation
        self.edge2faces.fill(-1)
        self.construct_egde_2_face()

        # Compute rest dihedral angle for bending
        self.compute_rest_dihedral_angle()

        # For visualization use
        self.indices = ti.field(dtype = ti.u32, shape = self.num_faces * 3)
        self.initIndices()
    

    def set_current_state_as_rest_state(self):
        # Set rest pose
        self._copy(self.verts.x, self.verts.rest_x)

        # Compute the rest length based on current rest state
        self.compute_rest_length()
        # Compute rest dihedral angle for bending
        self.compute_rest_dihedral_angle()
        self.clear_velocity()
    
    @ti.kernel
    def _copy(self, src: ti.template(), dst: ti.template()):
        for idx in src:
            dst[idx] = src[idx]

    @ti.kernel
    def clear_velocity(self):
        for idx in range(self.num_verts):
            self.verts[idx].v = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def compute_rest_length(self):
        for idx in range(self.num_edges):
            v1 = self.edges[idx].v1
            v2 = self.edges[idx].v2
            self.edges[idx].rest_len = (self.verts.x[v1] - self.verts.x[v2]).norm()

    @ti.kernel
    def compute_rest_dihedral_angle(self):
        # Compute rest dihedral angle
        for idx in range(self.num_edges):
            face_1 = self.edge2faces[idx, 0]
            face_2 = self.edge2faces[idx, 1]
            if face_1 != -1 and face_2 != -1:
                v1 = self.edges[idx].v1
                v2 = self.edges[idx].v2
                k, l = 0, 0
                
                for i in range(3):
                    if self.faces[face_1, i] != v1 and \
                        self.faces[face_1, i] != v2: k = i
                v3 = self.faces[face_1, k]

                for i in range(3):
                    if self.faces[face_2, i] != v1 and \
                        self.faces[face_2, i] != v2: l = i

                v4 = self.faces[face_2, l]
                
                w1, w2, w3, w4 = self.verts[v1].invM, self.verts[v2].invM, self.verts[v3].invM, self.verts[v4].invM
                if w1 + w2 + w3 + w4 > 0.:
                    # Appendix A: Bending Constraint Projection
                    p2 = self.verts[v2].new_x - self.verts[v1].new_x
                    p3 = self.verts[v3].new_x - self.verts[v1].new_x
                    p4 = self.verts[v4].new_x - self.verts[v1].new_x
                    l23 = p2.cross(p3).norm()
                    l24 = p2.cross(p4).norm()
                    if l23 < 1e-8: l23 = 1.
                    if l24 < 1e-8: l24 = 1.
                    n1 = p2.cross(p3) / l23
                    n2 = p2.cross(p4) / l24
                    d = ti.math.clamp(n1.dot(n2), -1., 1.)
                    self.rest_dihedral_angles[idx] = d


    @ti.kernel
    def construct_egde_2_face(self):
        for idx_i in range(self.num_edges):
            for idx_j in range(self.num_faces):
                # print(f"here")
                vertex_0 = self.edges[idx_i].v1
                vertex_1 = self.edges[idx_i].v2
                # print(f"v0: {vertex_0}, v1: {vertex_1} ")
                # print(f"v0, v1 ", vertex_0, vertex_1)
                if (vertex_0 == self.faces[idx_j, 0] or vertex_0 == self.faces[idx_j, 1] or vertex_0 == self.faces[idx_j, 2]) and (vertex_1 == self.faces[idx_j, 0] or vertex_1 == self.faces[idx_j, 1] or vertex_1 == self.faces[idx_j, 2]):
                    if self.edge2faces[idx_i, 0] == -1:
                        self.edge2faces[idx_i, 0] = idx_j
                    else:
                        self.edge2faces[idx_i, 1] = idx_j
        
        # # Compute rest dihedral angle
        # for idx in range(self.num_edges):
        #     face_1 = self.edge2faces[idx, 0]
        #     face_2 = self.edge2faces[idx, 1]
        #     if face_1 != -1 and face_2 != -1:
        #         v1 = self.edges[idx].v1
        #         v2 = self.edges[idx].v2
        #         k, l = 0, 0
                
        #         for i in range(3):
        #             if self.faces[face_1, i] != v1 and \
        #                 self.faces[face_1, i] != v2: k = i
        #         v3 = self.faces[face_1, k]

        #         for i in range(3):
        #             if self.faces[face_2, i] != v1 and \
        #                 self.faces[face_2, i] != v2: l = i

        #         v4 = self.faces[face_2, l]
                
        #         w1, w2, w3, w4 = self.verts[v1].invM, self.verts[v2].invM, self.verts[v3].invM, self.verts[v4].invM
        #         if w1 + w2 + w3 + w4 > 0.:
        #             # Appendix A: Bending Constraint Projection
        #             p2 = self.verts[v2].new_x - self.verts[v1].new_x
        #             p3 = self.verts[v3].new_x - self.verts[v1].new_x
        #             p4 = self.verts[v4].new_x - self.verts[v1].new_x
        #             l23 = p2.cross(p3).norm()
        #             l24 = p2.cross(p4).norm()
        #             if l23 < 1e-8: l23 = 1.
        #             if l24 < 1e-8: l24 = 1.
        #             n1 = p2.cross(p3) / l23
        #             n2 = p2.cross(p4) / l24
        #             d = ti.math.clamp(n1.dot(n2), -1., 1.)
        #             self.rest_dihedral_angles[idx] = d

    
    @ti.kernel
    def translation(self, offset: ti.math.vec3):
        for idx in range(self.num_verts):
            self.verts[idx].x += offset
    
    @ti.kernel
    def covert_sdf_color(self):
        max_val = 0.0
        min_val = 0.0
        for idx in self.verts.sdf_val:
            val = self.verts.sdf_val[idx]
            if val >= 0.0:
                ti.atomic_max(max_val, val)
            else:
                ti.atomic_min(min_val, val)

        for idx in self.verts.sdf_color:
            val = self.verts.sdf_val[idx]
            if val >= 0.0:
                sdf_val_normlized = val / max_val
                # Positive SDF, blue
                self.verts.sdf_color[idx] = ti.Vector([0.0, 0.0, sdf_val_normlized])
            else:
                sdf_val_normlized = val / min_val
                # Negative SDF, red
                self.verts.sdf_color[idx] = ti.Vector([sdf_val_normlized, 0.0, 0.0])


    @ti.kernel
    def initialize(self):
        for idx in range(self.num_verts):
            self.verts.x[idx] = self.verts.x[idx] * self.scale + ti.Vector(self.offset)
            self.verts.invM[idx] = self.mass

        for idx in range(self.num_edges):
            v1 = self.edges[idx].v1
            v2 = self.edges[idx].v2
            self.edges[idx].rest_len = (self.verts.x[v1] - self.verts.x[v2]).norm()
            # self.egdes.stretch_compliance
            self.edges[idx].stretch_compliance = self.stretch_compliance
            self.edges[idx].bending_compliance = self.bending_compliance

    @ti.kernel
    def initIndices(self):
        for idx in range(self.num_faces):
            self.indices[idx * 3 + 0] = self.faces[idx, 0]
            self.indices[idx * 3 + 1] = self.faces[idx, 1]
            self.indices[idx * 3 + 2] = self.faces[idx, 2]
    
    @ti.kernel
    def applyExtForce(self, dt : ti.f32):
        for idx in range(self.num_verts):
            if self.verts[idx].invM > 0.0:
                self.verts[idx].v = self.verts[idx].v + self.gravity[None] * dt
            self.verts[idx].new_x = self.verts[idx].x + self.verts[idx].v * dt
    
    @ti.kernel
    def update(self, dt : ti.f32):
        for idx in range(self.num_verts):
            if self.verts[idx].invM <= 0.0:
                self.verts[idx].new_x = self.verts[idx].x
            else:
                self.verts[idx].v = 1.0 * (self.verts[idx].new_x - self.verts[idx].x) / dt  # simple damping for reducing jittering
                self.verts[idx].x = self.verts[idx].new_x
    
    @ti.kernel
    def preSolve(self):
        self.verts.dp.fill(0.0)
    
    @ti.kernel
    def postSolve(self, sc : ti.template()):
        for idx in range(self.num_verts):
            self.verts[idx].new_x = self.verts[idx].new_x + self.verts[idx].dp * sc

    @ti.kernel
    def solveStretch(self, dt : ti.f32):
        ti.loop_config(block_dim=self.block_size)
        for idx in range(self.num_edges):
            v1 = self.edges[idx].v1
            v2 = self.edges[idx].v2
            w1, w2 = self.verts[v1].invM, self.verts[v2].invM
            if w1 + w2 > 0.:
                n = self.verts[v1].new_x - self.verts[v2].new_x
                d = n.norm()
                dp = ti.zero(n)
                constraint = (d - self.edges[idx].rest_len)
                if ti.static(self.XPBD): # https://matthias-research.github.io/pages/publications/XPBD.pdf
                    compliance = self.edges[idx].stretch_compliance / (dt**2)
                    # d_lambda = -(constraint + compliance * self.edges.lambda_stretch[idx]) / (w1 + w2 + compliance) * self.stretch_relaxation # eq. (18)
                    d_lambda = -(constraint) / (w1 + w2 + compliance) * self.stretch_relaxation # eq. (18)
                    dp = d_lambda * n.normalized(1e-12) # eq. (17)
                    self.edges[idx].lambda_stretch += d_lambda
                else: # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
                    dp = -constraint / (w1 + w2) * n.normalized(1e-12) * self.stretch_relaxation # eq. (1)
                self.verts[v1].dp += dp * w1
                self.verts[v2].dp -= dp * w2
        
        # for idx in range(self.num_verts):
        #     self.verts.new_x[idx] = self.verts.new_x[idx] + self.verts.dp[idx]
        #     self.verts.dp[idx].fill(0.0) 

    @ti.kernel
    def solveBending(self, dt : ti.f32):
        ti.loop_config(block_dim=self.block_size)
        for idx in range(self.num_edges):
            face_1 = self.edge2faces[idx, 0]
            face_2 = self.edge2faces[idx, 1]
            if face_1 != -1 and face_2 != -1:
                v1 = self.edges[idx].v1
                v2 = self.edges[idx].v2
                k, l = 0, 0
                
                for i in range(3):
                    if self.faces[face_1, i] != v1 and \
                        self.faces[face_1, i] != v2: k = i
                v3 = self.faces[face_1, k]


                for i in range(3):
                    if self.faces[face_2, i] != v1 and \
                        self.faces[face_2, i] != v2: l = i
                v4 = self.faces[face_2, l]
                
                w1, w2, w3, w4 = self.verts[v1].invM, self.verts[v2].invM, self.verts[v3].invM, self.verts[v4].invM
                if w1 + w2 + w3 + w4 > 0.:
                    # Appendix A: Bending Constraint Projection
                    p2 = self.verts[v2].new_x - self.verts[v1].new_x
                    p3 = self.verts[v3].new_x - self.verts[v1].new_x
                    p4 = self.verts[v4].new_x - self.verts[v1].new_x
                    l23 = p2.cross(p3).norm()
                    l24 = p2.cross(p4).norm()
                    if l23 < 1e-8: l23 = 1.
                    if l24 < 1e-8: l24 = 1.
                    n1 = p2.cross(p3) / l23
                    n2 = p2.cross(p4) / l24
                    d = ti.math.clamp(n1.dot(n2), -1., 1.)
                    
                    q3 = (p2.cross(n2) + n1.cross(p2) * d) / l23 # eq. (25)
                    q4 = (p2.cross(n1) + n2.cross(p2) * d) / l24 # eq. (26)
                    q2 = -(p3.cross(n2) + n1.cross(p3) * d) / l23 \
                            -(p4.cross(n1) + n2.cross(p4) * d) / l24 # eq. (27)
                    q1 = -q2 - q3 - q4
                    # eq. (29)
                    sum_wq = w1 * q1.norm_sqr() + \
                                w2 * q2.norm_sqr() + \
                                w3 * q3.norm_sqr() + \
                                w4 * q4.norm_sqr()
                    rest_d = -1.
                    rest_d = self.rest_dihedral_angles[idx]  # Get the rest dihedral angle between the two triangles
                    constraint = (ti.acos(d) - ti.acos(rest_d)) 
                    if ti.static(self.XPBD):
                        compliance = self.edges.bending_compliance[idx] / (dt**2)
                        # d_lambda = -(constraint + compliance * self.edges[idx].lambda_bending) / (sum_wq + compliance) * self.bending_relaxation # eq. (18)
                        d_lambda = -(constraint) / (sum_wq + compliance) * self.bending_relaxation # eq. (18)
                        constraint = ti.sqrt(1 - d ** 2) * d_lambda
                        self.edges[idx].lambda_bending += d_lambda
                    else:
                        constraint = -ti.sqrt(1 - d ** 2) * constraint / (sum_wq + 1e-7) * self.bending_relaxation
                    self.verts[v1].dp += w1 * constraint * q1
                    self.verts[v2].dp += w2 * constraint * q2
                    self.verts[v3].dp += w3 * constraint * q3
                    self.verts[v4].dp += w4 * constraint * q4
            else:
                # print("boundary edge ", idx, face_1, face_2)
                pass

        # for idx in range(self.num_verts):
        #     self.verts.new_x[idx] = self.verts.new_x[idx] + self.verts.dp[idx]
        #     self.verts.dp[idx].fill(0.0)
    
    @ti.kernel
    def update_position(self, step_size : ti.f32):
        for idx in range(self.num_verts):
            self.verts[idx].new_x = self.verts[idx].new_x + step_size * self.verts[idx].dp
            self.verts[idx].dp.fill(0.0)
    

    @ti.kernel
    def reorder_verts(self):
        self.sh.compute_reordered_idx(self.num_verts, self.verts.new_x, self.verts.reordered_idx)

        # copy to reordered
        for i in range(self.num_verts):
            reordered_idx = self.verts[i].reordered_idx
            self.verts_reordered[reordered_idx] = self.verts[i]

    
    @ti.kernel
    def copy_from_reordered(self):
        for i in range(self.num_verts):
            self.verts[i] = self.verts_reordered[self.verts[i].reordered_idx]
    

    @ti.func
    def _func_solve_self_collision(self, i, j):
        cur_dist = (self.verts_reordered[i].new_x - self.verts_reordered[j].new_x).norm(self.eps)
        rest_dist = (self.verts_reordered[i].rest_x - self.verts_reordered[j].rest_x).norm(self.eps)
        target_dist = self.particle_size # target particle distance is 2 * particle radius, i.e. particle_size
        if cur_dist < target_dist and rest_dist > target_dist:
            wi = self.verts_reordered[i].invM
            wj = self.verts_reordered[j].invM
            n = (self.verts_reordered[i].new_x - self.verts_reordered[j].new_x) / cur_dist

            ### resolve collision ###
            # self.verts_reordered[i].dpos += wi / (wi + wj) * (target_dist - cur_dist) * n
            self.verts_reordered[i].dp += wi / (wi + wj) * (target_dist - cur_dist) * n

            ### apply friction ###
            # https://mmacklin.com/uppfrta_preprint.pdf
            # equation (23)
            dv = (self.verts_reordered[i].new_x - self.verts_reordered[i].x) - (self.verts_reordered[j].new_x - self.verts_reordered[j].x)
            dpos = -(dv - n * n.dot(dv))
            # equation (24)
            d = target_dist - cur_dist
            mu_s = ti.max(self.verts_reordered[i].mu_s, self.verts_reordered[j].mu_s)
            mu_k = ti.max(self.verts_reordered[i].mu_k, self.verts_reordered[j].mu_k)
            if dpos.norm() < mu_s * d:
                self.verts_reordered.dp[i] += wi / (wi + wj) * dpos
            else:
                self.verts_reordered.dp[i] += wi / (wi + wj) * dpos * ti.min(1.0, mu_k * d / dpos.norm(self.eps))


    @ti.kernel
    def solve_self_collision(self):
        for i in range(self.num_verts):
            base = self.sh.pos_to_grid(self.verts[i].new_x)
            for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
                slot_idx = self.sh.grid_to_slot(base + offset)
                for j in range(self.sh.slot_start[slot_idx], self.sh.slot_size[slot_idx] + self.sh.slot_start[slot_idx]):
                    if i != j:
                        self._func_solve_self_collision(i, j)
        
        for i in range(self.num_verts):
            self.verts_reordered[i].new_x = self.verts_reordered[i].new_x + self.verts_reordered[i].dp
            self.verts_reordered[i].dp.fill(0)


    @ti.kernel
    def collision(self):
        for idx in range(self.num_verts):
            prev_x = self.verts[idx].x
            x = self.verts[idx].new_x
            diff = x - prev_x
            sdf_scale = self.sdf.scale

            n_iter_collision = 1
            thickness = 0.001 * 2 # TODO: This is a artificail surface layer to the SDF for better detection
            for k in range(n_iter_collision):
                x = prev_x + (k+1) / n_iter_collision * diff
                x_scaled = x * sdf_scale
                if x_scaled.x > 0.0 and x_scaled.x < 1.0 and x_scaled.y > 0.0 and x_scaled.y < 1.0 and x_scaled.z > 0.0 and x_scaled.z < 1.0:
                    nl = self.sdf.dist(x_scaled) - thickness 
                    # Record SDF values
                    self.verts[idx].sdf_val = nl
                    n = self.sdf.normal(x_scaled) #/ sdf_scale
                    if nl < 0:
                        # print(nl)
                        n = self.sdf.normal(x_scaled) #/ sdf_scale
                        if n.norm() < self.eps:
                            print(n, n.norm(), nl, x)
                            pass
                        else:
                            n = n.normalized()

                        n = -n
                        x = x + nl * n
                        self.verts[idx].delta_x += nl * n
                        
                        # Handling friction
                        x = self.verts[idx].delta_x + self.verts[idx].new_x - self.verts[idx].x
                        vel = x
                        n = n
                        vn = n.dot(vel) * n
                        vt = vel - vn
                        stress = ti.abs(nl) * 1.0
                        mu_s_i = 0.25 # TODO: magic number
                        mu_k_i = 0.2 # TODO: magic number
                        if vt.norm() < stress * mu_s_i:
                            x -= vt
                        else:
                            delta = vt * ti.min(
                                stress * mu_k_i  / vt.norm(), 1.0
                            )
                            x -= delta
                        self.verts[idx].delta_x = x - (self.verts[idx].new_x - self.verts[idx].x)
                        break
        for idx in range(self.num_verts):
            self.verts[idx].new_x = self.verts[idx].new_x +  1.0 / self.collision_iter_n * self.verts[idx].delta_x

    def solve(self):
        frame_time_left = self.frame_dt
        substep = 0
        while frame_time_left > 0.0:
            substep += 1
            dt0 = min(self.dt, frame_time_left)
            frame_time_left -= dt0

            self.applyExtForce(dt0)
            if self.XPBD:
                self.edges.lambda_stretch.fill(0.)
                self.edges.lambda_bending.fill(0.)
            for iter in range(self.rest_iter):
                self.preSolve()
                self.solveStretch(dt0)
                self.update_position(1.0)
                self.solveBending(dt0)
                self.update_position(1.0)
            self.verts.delta_x.fill(0.0)

            self.reorder_verts()
            self.solve_self_collision()
            self.copy_from_reordered()
            
            self.collision()
            self.update(dt0)

            self.time += dt0
    

    # def solve(self):
    #     dt0 = self.dt
    #     self.applyExtForce(dt0)
    #     if self.XPBD:
    #         self.edges.lambda_stretch.fill(0.)
    #         self.edges.lambda_bending.fill(0.)
    #     for iter in range(self.rest_iter):
    #         self.preSolve()
    #         self.solveStretch(dt0)
    #         self.update_position(1.0)
    #         self.solveBending(dt0)
    #         self.update_position(1.0)
    #         # self.postSolve(0.9)
    #     self.verts.delta_x.fill(0.0)

    #     self.reorder_verts()
    #     self.solve_self_collision()
    #     self.copy_from_reordered()
        
    #     self.collision()
    #     self.update(dt0)

        
    #     self.time += dt0
    

    def solve_by_step(self):
        dt0 = self.dt
        step_size = 0.1
        self.applyExtForce(dt0)
        if self.XPBD:
            self.edges.lambda_stretch.fill(0.)
            self.edges.lambda_bending.fill(0.)
        for iter in range(self.rest_iter):
            self.preSolve()
            self.solveStretch(dt0)
            self.update_position(step_size)
            self.solveBending(dt0)
            self.update_position(step_size)
            # self.postSolve(0.1)
        
        self.verts.delta_x.fill(0.0)
        self.collision()
        self.update(dt0)