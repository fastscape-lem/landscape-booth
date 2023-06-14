import numpy as np


class Particles:

    def __init__(self, toposim, scale, n_particles=13000, speed_factor=0.9):
        self.n_particles = n_particles
        self.toposim = toposim
        self.shape = toposim.shape
        self.scale = scale
        self.speed_factor = speed_factor

        self.sizes = np.random.randint(1, 4, n_particles)
        
    def _set_position(self, pidx=None):
        nrows, ncols = self.shape

        if pidx is None:
            size = self.n_particles
            self.row_idx = np.random.randint(1, nrows, size=size)
            self.col_idx = np.random.randint(0, ncols, size=size)
        else:
            size = pidx.size
            self.row_idx[pidx] = np.random.randint(1, nrows, size=size)
            self.col_idx[pidx] = np.random.randint(0, ncols, size=size)

    def initialize(self):
        self._set_position()

    def run_step(self):
        sel = self.toposim.receivers[self.row_idx, self.col_idx].ravel()

        # slower the evolution, particles don't always move
        n = int(self.n_particles * self.speed_factor)
        pidx = np.arange(self.n_particles)
        np.random.shuffle(pidx)
        move_idx = pidx[0:n]

        self.row_idx[move_idx] = sel[move_idx] % self.shape[0]
        self.col_idx[move_idx] = ((sel[move_idx] - self.row_idx[move_idx])
                                  // (self.shape[0]))
        
        # re-distribute particles that reach the base levels within the domain
        at_base_level = np.argwhere(
            (self.row_idx == 0) | (self.row_idx == self.shape[0] - 1) | (self.col_idx == 0) | (self.col_idx == self.shape[1] - 1)
        )
        self._set_position(at_base_level.ravel())

    @property
    def positions(self):
        x = self.col_idx * self.scale
        y = self.row_idx * self.scale

        return x.astype(np.int32), y.astype(np.int32)

    def reset(self):
        self.initialize()
