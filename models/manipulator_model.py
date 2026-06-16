import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.1, r3=0.05):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2

        self.alpha = (
            self.m1 * self.d1**2
            + self.I_1
            + self.m2 * (self.l1**2 + self.d2**2)
            + self.I_2
            + self.m3 * (self.l1**2 + self.l2**2)
            + self.I_3
        )

        self.beta = (
            self.m2 * self.l1 * self.d2
            + self.m3 * self.l1 * self.l2
            )

        self.gamma = (
            self.m2 * self.d2**2
            + self.I_2
            + self.m3 * self.l2**2
            + self.I_3
            )

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        M_1_1 = self.alpha + 2 * self.beta * np.cos(q2)
        M_1_2 = self.gamma + self.beta * np.cos(q2)
        M_2_1 = self.gamma + self.beta * np.cos(q2)
        M_2_2 = self.gamma

        return np.array([[M_1_1, M_1_2], [M_2_1, M_2_2]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        C_1_1 = -self.beta * np.sin(q2) * q2_dot
        C_1_2 = -self.beta * np.sin(q2) * (q1_dot + q2_dot)
        C_2_1 =  self.beta * np.sin(q2) * q1_dot
        C_2_2 = 0

        return np.array([[C_1_1, C_1_2], [C_2_1, C_2_2]])

    def x_dot(self, x, u):
        """
        State space form of the dynamics (21): x_dot = A @ x + B @ u, where the
        upper block is just velocities and the lower block comes from
        q_ddot = M^-1 (u - C q_dot). Used by the MMAC to predict the next state.
        """
        x = np.array(x).flatten()
        M_inv = np.linalg.inv(self.M(x))
        C = self.C(x)

        zeros = np.zeros((2, 2))
        A = np.block([
            [zeros, np.eye(2)],
            [zeros, -M_inv @ C],
        ])
        B = np.concatenate([zeros, M_inv], axis=0)

        return A @ x.reshape(4, 1) + B @ np.array(u).reshape(2, 1)

