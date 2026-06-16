from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        z = self.state.reshape(-1, 1)
        y = np.atleast_2d(q).reshape(-1, 1)
        u = np.atleast_2d(u).reshape(-1, 1)
        z_dot = self.A @ z + self.B @ u + self.L @ (y - self.W @ z)
        self.state = (z + self.Tp * z_dot).flatten()

    def get_state(self):
        return self.state
