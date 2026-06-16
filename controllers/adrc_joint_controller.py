import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])
        B = np.array([[0], [b], [0]])
        L = np.array([[3 * p], [3 * p ** 2], [p ** 3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

        self.last_u = 0.0

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q, _ = x

        self.eso.update(q, self.last_u)
        _, q_dot_est, f_est = self.eso.get_state()

        v = q_d_ddot + self.kp * (q_d - q) + self.kd * (q_d_dot - q_dot_est)
        u = (v - f_est) / self.b

        self.last_u = u
        return u
