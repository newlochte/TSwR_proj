import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp, m3=0.1, r3=0.05)
        self.Kp = Kp
        self.Kd = Kd
        self.last_u = np.zeros((2, 1))

        self.L = np.array([
            [3 * p[0], 0],
            [0, 3 * p[1]],
            [3 * p[0] ** 2, 0],
            [0, 3 * p[1] ** 2],
            [p[0] ** 3, 0],
            [0, p[1] ** 3],
        ])

        W = np.zeros((2, 6))
        W[0:2, 0:2] = np.eye(2)

        A = np.zeros((6, 6))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 4:6] = np.eye(2)

        B = np.zeros((6, 2))

        self.eso = ESO(A, B, W, self.L, q0, Tp)

    def update_params(self, M, C):
        ### TODO Implement procedure to set eso.A and eso.B
        M_inv = np.linalg.inv(M)

        A = np.zeros((6, 6))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C
        A[2:4, 4:6] = np.eye(2)

        B = np.zeros((6, 2))
        B[2:4, :] = M_inv

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q1, q2, _, _ = x
        q = np.array([[q1], [q2]])

        M = self.model.M(x)
        C = self.model.C(x)

        self.update_params(M, C)
        self.eso.update(q, self.last_u)

        _, _, q1_dot_est, q2_dot_est, f1_est, f2_est = self.eso.get_state()
        q_dot_est = np.array([[q1_dot_est], [q2_dot_est]])
        f_est = np.array([[f1_est], [f2_est]])

        v = (
            q_d_ddot.reshape(2, 1)
            + self.Kd @ (q_d_dot.reshape(2, 1) - q_dot_est)
            + self.Kp @ (q_d.reshape(2, 1) - q)
        )
        u = M @ (v - f_est) + C @ q_dot_est

        self.last_u = u
        return u.flatten()
