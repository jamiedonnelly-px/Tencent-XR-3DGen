import taichi as ti

vec3 = ti.types.vector(3, ti.f32)

class Visualizer:
    def __init__(self) -> None:
        self.window = None
        self.assemble_reference_box()
        self.show_particle = True
        self.show_sdf = False
        self.cloth_fit = False
        self.run_sim = False
        self.ready_for_sim = False # Set the current state as the initial state for sim
        self.export_obj = False

    def initScene(self, position, lookat, show_window=True):
        global __show_window
        __show_window = show_window

        global scene, window, camera, canvas
        window = ti.ui.Window("XPBD Cloth", (1024, 768), show_window=show_window, vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(position[0], position[1], position[2])
        camera.up(0, 0, 1.0)
        camera.lookat(lookat[0], lookat[1], lookat[2])
        camera.fov(55)
        self.window = window
        return window

    # A [0, 1]^3 unit cube for reference
    def assemble_reference_box(self):
        self.reference_box_lines = ti.Vector.field(3, ti.f32, 12*2)
        line_cnt = 0
        for i in range(8):
            # Enumerate all vertices
            mask = []
            for j in range(3):
                mask.append((i >> j) & 1)
            vertex_a = vec3(
                mask[0],
                mask[1],
                mask[2]
            )
            # Enumerate all neighbor vertices of this vertex
            for j in range(3):
                vertex_b = vec3(vertex_a.x, vertex_a.y, vertex_a.z)
                # Avoid duplicated edges
                if mask[j] ^ 1 == 1:
                    vertex_b[j] = mask[j] ^ 1
                    self.reference_box_lines[line_cnt * 2] = vertex_a
                    self.reference_box_lines[line_cnt * 2 + 1] = vertex_b
                    line_cnt += 1


    def renderScene(self, solver, smpl_mesh, frame):
        global scene, window, camera, canvas
        global __show_window
        camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        self.show_particle = self.window.GUI.checkbox("show_particle", self.show_particle)
        self.cloth_fit = self.window.GUI.checkbox("cloth_fit", self.cloth_fit)
        self.ready_for_sim = self.window.GUI.checkbox("ready_for_sim", self.ready_for_sim)
        self.run_sim = self.window.GUI.checkbox("run_sim", self.run_sim)
        self.export_obj = self.window.GUI.checkbox("export_obj", self.export_obj)

        solver.sdf.render(scene)
        scene.mesh(solver.verts.x, solver.indices, color=(0.5, 0.5, 0.5), two_sided=True)
        if self.show_particle:
            scene.particles(solver.verts.x, 0.0025, per_vertex_color=solver.verts.sdf_color)
        scene.lines(self.reference_box_lines, color = (0.0, 1.0, 0.0), width = 1.0)

        canvas.scene(scene)
        if __show_window:
            window.show()
        else:
            window.save_image(f"results/frame/{frame:06d}.jpg")
        # for event in window.get_events(ti.ui.PRESS):
        #     if event.key in [ti.ui.ESCAPE]:
        #         window.running = False
        return window.running

    def exportScene(self, solver, frame, output):
        x_np = solver.mesh.verts.x.to_numpy()
        indices_np = solver.indices.to_numpy().reshape(-1, 3)

        with open(f'{output}/model_{frame}.obj', "w") as output:
            for i in range(len(solver.mesh.verts)):
                output.write(f"v {x_np[i, 0]} {x_np[i, 1]} {x_np[i, 2]}\n")
            for i in range(len (solver.mesh.faces)):
                output.write(f"f {indices_np[i, 0]+1} {indices_np[i, 1]+1} {indices_np[i, 2]+1}\n")