import taichi as ti
import numpy as np


@ti.data_oriented
class SpatialHasher:
    def __init__(self, cell_size, grid_res, n_slots=None):
        self.cell_size = cell_size
        self.grid_res = grid_res

        if n_slots is None:
            self.n_slots = np.prod(grid_res)
        else:
            self.n_slots = n_slots
    
    def build(self):
        # number of elements in each slot
        self.slot_size  = ti.field(ti.int32, shape=self.n_slots)
        # element index offset in each slot
        self.slot_start = ti.field(ti.int32, shape=self.n_slots)


    @ti.func
    def compute_reordered_idx(self, n, pos, reordered_idx):
        """
        Reordered element idx based on the given positions and active flags.

        Parameters:
            n (int)       : The number of elements in the positions and active arrays.
            pos           : The array of positions.
            reordered_idx : The array to store the computed reordered indices.

        Returns:
            None
        """

        self.slot_size.fill(0)
        self.slot_start.fill(0)

        for i in range(n):
            slot_idx = self.pos_to_slot(pos[i])
            ti.atomic_add(self.slot_size[slot_idx], 1)

        cur_cnt = 0
        for i in range(self.n_slots):
            self.slot_start[i] = ti.atomic_add(cur_cnt, self.slot_size[i])

        for i in range(n):
            slot_idx = self.pos_to_slot(pos[i])
            reordered_idx[i] = ti.atomic_add(self.slot_start[slot_idx], 1)

        # recover slot_start 
        for i in range(self.n_slots):
            self.slot_start[i] -= self.slot_size[i]


    @ti.func
    def for_all_neighbors(
        self,
        i,
        pos,
        task_range,
        ret  : ti.template(),
        task : ti.template(),
    ):
        """
        Iterates over all neighbors of a given position and performs a task on each neighbor.
        Elements are considered neighbors if they are within task_range.

        Parameters:
            i (int)    : Index of the querying particle.
            pos        : Template for the positions of all particles.
            task       : Template for the task to be performed on each neighbor of the querying particle.
            task_range : Range within which the task should be performed.
            ret        : Template for the return value of the task.

        Returns:
            None
        """
        base = self.pos_to_grid(pos[i])
        for offset in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
            slot_idx = self.grid_to_slot(base + offset)
            for j in range(self.slot_start[slot_idx], self.slot_size[slot_idx] + self.slot_start[slot_idx]):
                if i != j and (pos[i] - pos[j]).norm() < task_range:
                    task(i, j, ret)

    @ti.func
    def pos_to_grid(self, pos):
        return ti.floor(pos / self.cell_size, ti.int32)

    @ti.func
    def grid_to_pos(self, grid_id):
        return (grid_id + 0.5 ) * self.cell_size

    @ti.func
    def grid_to_slot(self, grid_id):
        return (grid_id[0] * self.grid_res[1] * self.grid_res[2] + grid_id[1] * self.grid_res[2] + grid_id[2]) % self.n_slots
    
    @ti.func
    def pos_to_slot(self, pos):
        return self.grid_to_slot(self.pos_to_grid(pos))